import os
import atexit
import struct
import subprocess
import tempfile
import threading
import queue
import time

import numpy as np
import trimesh

# Flow panel wrapper, credit to Claude Opus 4.8. It starts a Julia server to
# handle communication with FP.jl (implementation also credited to Claude Opus
# 4.8 to make use of the FLOWPanel.jl library, credit to Eduardo J. Alvarez
# (main) and Cibin Joseph)


class FLOWPanelSolver:
    """
    Python wrapper around the Julia panel solver in FP.jl.

    Coordinate convention (unchanged): mesh and query points must already be in
    the same coordinate system before this class is constructed. This wrapper
    does NOT secretly scale query points.

    Performance: launches ONE persistent Julia process that loads the mesh and
    solves the panel system a single time, then streams query-point batches over
    stdin/stdout. Previously each velocity() call spawned a fresh Julia process
    (paying interpreter startup + JIT + re-solve every time).

    Portability: the startup-readiness wait reads Julia's stderr on a background
    thread instead of select.select(). select() does not work on pipe handles on
    Windows (it accepts sockets only), which is why the previous version raised a
    socket error there. The thread-based pump works identically on Windows, WSL,
    and native Linux, and also keeps draining stderr for the life of the process
    so Julia diagnostics can never fill the pipe buffer and deadlock the solver.
    """

    def __init__(
        self,
        mesh,
        V_inf,
        julia_script="FP.jl",
        julia_bin="julia",
        verbose=True,
        max_julia_points=2_000_000,
        startup_timeout=120.0,
    ):
        if isinstance(mesh, str):
            self.mesh = trimesh.load_mesh(mesh)
        else:
            self.mesh = mesh.copy()

        self.V_inf = np.asarray(V_inf, dtype=float)
        self._julia_bin = julia_bin
        self._julia_script = julia_script
        self._verbose = verbose
        # Cap on points per frame, to bound Julia-side memory. Frames reuse the
        # same live process, so this is NOT per-process chunking — no startup
        # cost between frames.
        self._max_julia_points = int(max_julia_points)
        # Generous by default: the first server start pays full package import +
        # JIT compilation, which can take tens of seconds on a cold Julia.
        self._startup_timeout = float(startup_timeout)

        self.diag = float(np.linalg.norm(self.mesh.extents))

        # Write mesh to temp STL so Julia can read it.
        self._tmpdir = tempfile.mkdtemp()
        self.stl_path = os.path.join(self._tmpdir, "body.stl")
        self.mesh.export(self.stl_path)

        self._proc = None
        self._stderr_thread = None
        self._stderr_lines = None
        self._start_server()
        atexit.register(self.close)

        if self._verbose:
            print("\nFLOWPanelSolver initialized (persistent server)")
            print("-----------------------------------------------")
            print(f"Julia binary:  {self._julia_bin}")
            print(f"Julia script:  {self._julia_script}")
            print(f"Temp STL path: {self.stl_path}")
            print(f"V_inf:         {self.V_inf}")
            print(f"Mesh bounds:\n{self.mesh.bounds}")
            print(f"Mesh extents:  {self.mesh.extents}")
            print(f"Mesh diag:     {self.diag}")

    # ------------------------------------------------------------------ server
    def _start_server(self):
        cmd = [
            self._julia_bin,
            "--project=.",
            self._julia_script,
            self.stl_path,
            *[str(v) for v in self.V_inf],
            "--server",
        ]
        # stdout is a clean binary channel; Julia prints diagnostics to stderr.
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        # Drain stderr on a background thread so reading it never blocks the main
        # thread and never uses select() (which doesn't work on Windows pipes).
        # The thread runs for the life of the process so Julia diagnostics don't
        # fill and stall the stderr pipe buffer mid-evaluation.
        self._stderr_lines = queue.Queue()

        def _pump_stderr():
            for raw in iter(self._proc.stderr.readline, b""):
                text = raw.decode(errors="replace").rstrip()
                self._stderr_lines.put(text)
                if self._verbose and text:
                    print(f"[julia] {text}", flush=True)

        self._stderr_thread = threading.Thread(target=_pump_stderr, daemon=True)
        self._stderr_thread.start()

        # Wait for the readiness banner, but bail out if Julia dies or hangs.
        deadline = time.monotonic() + self._startup_timeout
        collected = []
        ready = False
        while time.monotonic() < deadline:
            try:
                text = self._stderr_lines.get(timeout=0.5)
                collected.append(text)
                if "FP.jl server ready" in text:
                    ready = True
                    break
            except queue.Empty:
                if self._proc.poll() is not None:
                    break  # process exited before signalling ready

        if not ready:
            rc = self._proc.poll()
            # Drain anything else the pump captured before raising.
            while True:
                try:
                    collected.append(self._stderr_lines.get_nowait())
                except queue.Empty:
                    break
            raise RuntimeError(
                f"Julia server failed to start (exit code {rc}).\n"
                f"Command:\n{' '.join(cmd)}\n"
                f"--- stderr ---\n" + "\n".join(collected)
            )

    def _drain_stderr(self):
        """Collect whatever stderr text the pump has buffered (non-blocking)."""
        lines = []
        if self._stderr_lines is None:
            return ""
        while True:
            try:
                lines.append(self._stderr_lines.get_nowait())
            except queue.Empty:
                break
        return "\n".join(lines)

    def _read_exact(self, n):
        """Read exactly n bytes from the Julia stdout pipe or raise."""
        buf = bytearray()
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                rc = self._proc.poll()
                err = self._drain_stderr()
                raise RuntimeError(
                    f"Julia server closed pipe mid-read (exit code {rc}).\n"
                    f"--- stderr ---\n{err}"
                )
            buf.extend(chunk)
        return bytes(buf)

    def _eval_frame(self, pts):
        """Send one (M,3) frame to the live Julia process, get (M,3) back."""
        M = pts.shape[0]
        flat = np.ascontiguousarray(pts, dtype=np.float64).reshape(-1)  # x0,y0,z0,...
        self._proc.stdin.write(struct.pack("<q", M))
        self._proc.stdin.write(flat.tobytes())
        self._proc.stdin.flush()

        n_back = struct.unpack("<q", self._read_exact(8))[0]
        if n_back != M:
            raise RuntimeError(f"Julia returned {n_back} points, expected {M}")
        raw = self._read_exact(M * 3 * 8)
        return np.frombuffer(raw, dtype=np.float64).reshape(M, 3).copy()

    def _call_julia(self, pts):
        """
        Evaluate velocity at pts via the persistent server.

        pts must be shape (N, 3) and in the same units/coordinates as self.mesh.
        Splits into frames bounded by max_julia_points purely to cap Julia-side
        memory; frames reuse the same process (no per-frame startup cost).
        """
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"pts must have shape (N, 3), got {pts.shape}")
        if np.isnan(pts).any():
            raise ValueError("NaNs found in query points passed to Julia.")
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("Julia server is not running.")

        N = pts.shape[0]
        step = self._max_julia_points
        if N <= step:
            return self._eval_frame(pts)

        out = np.empty((N, 3), dtype=np.float64)
        for i in range(0, N, step):
            out[i:i + step] = self._eval_frame(pts[i:i + step])
        return out

    def close(self):
        """Shut down the Julia server (idempotent)."""
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                # N <= 0 sentinel tells the server to exit its loop.
                self._proc.stdin.write(struct.pack("<q", 0))
                self._proc.stdin.flush()
                self._proc.stdin.close()
                self._proc.wait(timeout=10)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            self._proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # --------------------------------------------------------------- public API
    def velocity(self, pts, blank_interior=True, blank_near=False, near_tol=None, **_):
        """
        Evaluate velocity at arbitrary points.

        pts:            shape (N, 3)
        blank_interior: replace velocities inside the mesh with NaN.
        blank_near:     replace velocities within near_tol of the surface with NaN.
        """
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"pts must have shape (N, 3), got {pts.shape}")

        vel = self._call_julia(pts)

        if blank_interior:
            try:
                inside = self.mesh.contains(pts)
                vel[inside] = np.nan
            except Exception as exc:
                if self._verbose:
                    print(f"mesh.contains() failed in velocity(): {exc}")

        if blank_near and near_tol is not None and near_tol > 0:
            try:
                query = trimesh.proximity.ProximityQuery(self.mesh)
                signed_dist = query.signed_distance(pts)
                near = np.abs(signed_dist) < near_tol
                vel[near] = np.nan
            except Exception as exc:
                if self._verbose:
                    print(f"ProximityQuery failed in velocity(): {exc}")

        return vel

    def generate_flow_field(self, x, y, z, zero_inside=True):
        """
        Evaluate velocity on a meshgrid. Returns flattened (N, 3).
        """
        pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        vel = self._call_julia(pts)
        if zero_inside:
            try:
                inside = self.mesh.contains(pts)
                vel[inside] = 0.0
            except Exception as exc:
                if self._verbose:
                    print(f"mesh.contains() failed in generate_flow_field(): {exc}")
        return vel

    # Compatibility stubs.
    def bc_residual(self):
        return float("nan")

    def net_source(self):
        return float("nan")

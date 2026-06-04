import os
import sys
import struct
import atexit
import shutil
import threading
import subprocess
import tempfile

import numpy as np
import trimesh


# Binary protocol constants (must match FP.jl). All little-endian on the wire.
_MAGIC_QUERY = b"FPQ1"
_MAGIC_QUIT = b"FPXX"
_STATUS_OK = 0x00
_STATUS_ERR = 0x01


class FLOWPanelSolver:
    """
    Python wrapper around the Julia panel solver in FP.jl.

    Speed model:
        Julia + FLOWPanel boot and the panel solve are EXPENSIVE and happen
        ONCE, when this object is constructed. After that, every velocity()
        call streams its query points to a long-lived Julia process over pipes
        and reads results back. No per-call Julia startup, no per-call panel
        solve.

    Safety model:
        If the persistent server cannot be started (Julia missing, package
        env broken, pipe failure, protocol desync, ...) the wrapper transparently
        falls back to the original one-shot-per-call mode using temp .npy files.
        Results are identical; only speed differs. Set FLOWPANEL_FORCE_ONESHOT=1
        to disable the server entirely.

    Cross-platform notes (why this works the same on Windows and Linux):
        - The Julia project is resolved to an ABSOLUTE path next to FP.jl, so it
          no longer depends on the current working directory ("--project=." was
          the main Windows footgun).
        - All child-process text is decoded as UTF-8 with errors="replace", so
          Julia's UTF-8 log output never trips the Windows locale codec.
        - The data channel is a length-delimited little-endian binary protocol,
          so byte order and buffering behave identically on every OS. stdout is
          binary-only; all Julia logging is routed to stderr.

    Coordinate convention (unchanged):
        The mesh and all query points must already be in the same coordinate
        system. This wrapper does NOT secretly scale query points.
    """

    def __init__(
        self,
        mesh,
        V_inf,
        julia_script="FP.jl",
        julia_bin="julia",
        project=None,
        verbose=True,
    ):
        # Accept either a trimesh object or a path string.
        if isinstance(mesh, str):
            self.mesh = trimesh.load_mesh(mesh)
        else:
            self.mesh = mesh.copy()

        self.V_inf = np.asarray(V_inf, dtype=float).reshape(3)
        self._julia_bin = julia_bin
        # Resolve the script to an absolute path so CWD never matters.
        self._julia_script = os.path.abspath(julia_script)
        self._verbose = verbose

        # Project directory defaults to the folder containing FP.jl, absolute.
        if project is None:
            project = os.path.dirname(self._julia_script) or os.getcwd()
        self._project = os.path.abspath(project)

        self.diag = float(np.linalg.norm(self.mesh.extents))

        # Persistent STL on disk for the whole lifetime of this solver.
        # (Each call previously rewrote a temp STL via the mesh; we keep one.)
        self._tmpdir = tempfile.mkdtemp(prefix="flowpanel_")
        self.stl_path = os.path.join(self._tmpdir, "body.stl")
        self.mesh.export(self.stl_path)

        self._proc = None
        self._stderr_thread = None
        self._lock = threading.Lock()
        self._npanels = None
        self._mode = "oneshot"  # becomes "server" if startup succeeds

        atexit.register(self.close)

        if self._verbose:
            print("\nFLOWPanelSolver initialized")
            print("---------------------------")
            print(f"Julia binary:  {self._julia_bin}")
            print(f"Julia script:  {self._julia_script}")
            print(f"Julia project: {self._project}")
            print(f"Temp STL path: {self.stl_path}")
            print(f"V_inf:         {self.V_inf}")
            print(f"Mesh bounds:\n{self.mesh.bounds}")
            print(f"Mesh extents:  {self.mesh.extents}")
            print(f"Mesh diag:     {self.diag}")

        force_oneshot = os.environ.get("FLOWPANEL_FORCE_ONESHOT", "0") == "1"
        if not force_oneshot:
            try:
                self._start_server()
                self._mode = "server"
                if self._verbose:
                    print(f"Julia server ready (panels={self._npanels})")
            except Exception as exc:
                if self._verbose:
                    print(f"Server startup failed, falling back to one-shot: {exc}")
                self._kill_proc()
                self._mode = "oneshot"

    # ------------------------------------------------------------------ #
    # Server lifecycle
    # ------------------------------------------------------------------ #

    def _common_cmd_prefix(self):
        return [self._julia_bin, f"--project={self._project}", self._julia_script]

    def _start_server(self):
        cmd = self._common_cmd_prefix() + [
            "--server",
            self.stl_path,
        ]

        # On Windows, avoid inheriting a console that could interfere; on POSIX
        # start a new session so the child doesn't catch our signals.
        popen_kwargs = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,   # binary protocol
            stderr=subprocess.PIPE,   # text logs
            bufsize=0,                # unbuffered binary pipes
            cwd=self._project,
        )
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True

        self._proc = subprocess.Popen(cmd, **popen_kwargs)

        # Drain stderr in a background thread so it can't fill the pipe and
        # block Julia. Decode UTF-8 leniently (Windows-safe).
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

        # Hand the freestream over as binary BEFORE anything else. Forced
        # little-endian Float64 x3 -> locale-proof, byte-order-proof.
        vinf = np.ascontiguousarray(self.V_inf, dtype="<f8").reshape(3)
        self._proc.stdin.write(vinf.tobytes())
        self._proc.stdin.flush()

        # Wait for the single "READY <n>\n" line on stdout. Read byte-by-byte
        # until newline so we don't consume any following binary bytes.
        line = self._read_ready_line(timeout_bytes_max=4096)
        if not line.startswith("READY"):
            raise RuntimeError(f"server did not report READY; got: {line!r}")
        try:
            self._npanels = int(line.split()[1])
        except Exception:
            self._npanels = None

    def _read_ready_line(self, timeout_bytes_max=4096):
        """Read one '\\n'-terminated line from stdout without over-reading."""
        out = self._proc.stdout
        chars = bytearray()
        for _ in range(timeout_bytes_max):
            b = out.read(1)
            if b == b"":
                # Process died before READY.
                rc = self._proc.poll()
                raise RuntimeError(
                    f"Julia exited before READY (returncode={rc}). "
                    f"Check that '{self._julia_bin}' runs and the project at "
                    f"'{self._project}' is instantiated."
                )
            if b == b"\n":
                break
            chars.extend(b)
        return chars.decode("utf-8", errors="replace").strip()

    def _drain_stderr(self):
        try:
            for raw in iter(self._proc.stderr.readline, b""):
                if self._verbose:
                    txt = raw.decode("utf-8", errors="replace").rstrip()
                    if txt:
                        print(f"[julia] {txt}")
        except Exception:
            pass

    def _kill_proc(self):
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                try:
                    self._proc.stdin.write(_MAGIC_QUIT)
                    self._proc.stdin.flush()
                    self._proc.stdin.close()
                except Exception:
                    pass
                try:
                    self._proc.wait(timeout=5)
                except Exception:
                    self._proc.kill()
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            self._proc = None

    def close(self):
        """Shut the Julia server down and remove temp files. Idempotent."""
        self._kill_proc()
        tmp = getattr(self, "_tmpdir", None)
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
            self._tmpdir = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Core evaluation
    # ------------------------------------------------------------------ #

    def _eval_points(self, pts):
        """Dispatch to the server, falling back to one-shot on any failure."""
        if self._mode == "server" and self._proc is not None:
            try:
                return self._eval_server(pts)
            except Exception as exc:
                if self._verbose:
                    print(f"Server eval failed ({exc}); reverting to one-shot.")
                self._kill_proc()
                self._mode = "oneshot"
        return self._eval_oneshot(pts)

    def _eval_server(self, pts):
        """Send one query frame, read one response frame. Thread-safe."""
        n = pts.shape[0]
        # point-major flat float64, forced little-endian
        flat = np.ascontiguousarray(pts, dtype="<f8").reshape(-1)

        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                raise RuntimeError("Julia server process is not alive.")

            stdin = self._proc.stdin
            stdout = self._proc.stdout

            stdin.write(_MAGIC_QUERY)
            stdin.write(struct.pack("<q", n))
            stdin.write(flat.tobytes())
            stdin.flush()

            status = self._read_n(stdout, 1)
            if status[0] == _STATUS_ERR:
                msglen = struct.unpack("<q", self._read_n(stdout, 8))[0]
                msg = self._read_n(stdout, msglen).decode("utf-8", errors="replace")
                raise RuntimeError(f"Julia server error: {msg}")
            elif status[0] != _STATUS_OK:
                raise RuntimeError(f"Unknown status byte from Julia: {status[0]}")

            n_out = struct.unpack("<q", self._read_n(stdout, 8))[0]
            if n_out != n:
                raise RuntimeError(f"server returned {n_out} points, expected {n}")
            payload = self._read_n(stdout, 8 * 3 * n)

        vel = np.frombuffer(payload, dtype="<f8").astype(np.float64).reshape(n, 3)
        return np.array(vel)  # own the buffer

    @staticmethod
    def _read_n(stream, n):
        """Read exactly n bytes or raise (pipes can return short reads)."""
        if n == 0:
            return b""
        buf = bytearray()
        while len(buf) < n:
            chunk = stream.read(n - len(buf))
            if not chunk:
                raise RuntimeError("pipe closed mid-read (Julia died?)")
            buf.extend(chunk)
        return bytes(buf)

    def _eval_oneshot(self, pts):
        """Original behavior: boot Julia, file handshake, exit. One call."""
        with tempfile.TemporaryDirectory() as tmp:
            vinf_file = os.path.join(tmp, "vinf.npy")
            pts_file = os.path.join(tmp, "pts.npy")
            out_file = os.path.join(tmp, "vel.npy")
            # Freestream as binary .npy, not as command-line text: locale-proof.
            np.save(vinf_file, self.V_inf.astype(np.float64))
            np.save(pts_file, pts.astype(np.float64))

            cmd = self._common_cmd_prefix() + [
                self.stl_path,
                vinf_file,
                pts_file,
                out_file,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self._project,
            )

            if self._verbose and result.stdout:
                print(result.stdout)

            if result.returncode != 0:
                raise RuntimeError(
                    f"Julia failed with exit code {result.returncode}.\n"
                    f"Command:\n{' '.join(cmd)}\n\n"
                    f"--- stdout ---\n{result.stdout}\n\n"
                    f"--- stderr ---\n{result.stderr}"
                )
            if not os.path.exists(out_file):
                raise RuntimeError(
                    f"Julia finished but output file was not created:\n{out_file}\n"
                    f"--- stderr ---\n{result.stderr}"
                )

            vel = np.load(out_file)

        vel = np.asarray(vel, dtype=float)
        if vel.shape != pts.shape:
            raise RuntimeError(
                f"Julia returned velocity shape {vel.shape}, expected {pts.shape}"
            )
        return vel

    # ------------------------------------------------------------------ #
    # Public API (unchanged signatures)
    # ------------------------------------------------------------------ #

    def velocity(self, pts, blank_interior=True, blank_near=False, near_tol=None, **_):
        """
        Evaluate velocity at arbitrary points.

        pts:            shape (N, 3)
        blank_interior: if True, points inside the mesh get NaN.
        blank_near:     if True and near_tol > 0, points within near_tol of the
                        surface get NaN (uses trimesh signed distance).
        """
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"pts must have shape (N, 3), got {pts.shape}")
        if np.isnan(pts).any():
            raise ValueError("NaNs found in query points passed to the solver.")

        vel = self._eval_points(pts)

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
        vel = self._eval_points(pts)

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
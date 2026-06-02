# flowpanel_wrapper.py
import subprocess, tempfile, os
import numpy as np
import trimesh

class FLOWPanelSolver:
    """
    Drop-in replacement for VortexSheetSolver.
    Delegates to Julia/FP.jl via subprocess.
    """
    def __init__(self, mesh, V_inf,
                 julia_script="FP.jl",
                 julia_bin="julia",
                 verbose=True):

        # accept either a trimesh object or a path string
        if isinstance(mesh, str):
            self.stl_path = mesh
            self.mesh = trimesh.load_mesh(mesh)
        else:
            self.mesh = mesh
            # write to a temp STL so Julia can read it
            self._tmpdir  = tempfile.mkdtemp()
            self.stl_path = os.path.join(self._tmpdir, "body.stl")
            mesh.export(self.stl_path)

        self.V_inf        = np.asarray(V_inf, dtype=float)
        self._julia_bin   = julia_bin
        self._julia_script = julia_script
        self._verbose      = verbose
        self.diag          = float(np.linalg.norm(self.mesh.extents))

    def _call_julia(self, pts: np.ndarray) -> np.ndarray:
        """Write pts → npy, call Julia, read result npy."""
        with tempfile.TemporaryDirectory() as tmp:
            pts_file = os.path.join(tmp, "pts.npy")
            out_file = os.path.join(tmp, "vel.npy")
            np.save(pts_file, pts.astype(np.float64))

            cmd = [
                self._julia_bin,
                "--project=.",        
                self._julia_script,
                self.stl_path,
                *[str(v) for v in self.V_inf],
                pts_file, out_file,
            ]
            
            # always capture so the error is never thrown away; stream stdout
            # ourselves when verbose
            result = subprocess.run(cmd, capture_output=True, text=True)
            if self._verbose:
                if result.stdout:
                    print(result.stdout)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Julia failed (exit {result.returncode}):\n"
                    f"--- stdout ---\n{result.stdout}\n"
                    f"--- stderr ---\n{result.stderr}"
                )
            return np.load(out_file)   # (N, 3)

    # ---------- public interface (matches VortexSheetSolver) ----------

    def velocity(self, pts, blank_interior=True, blank_near=True, **_):
        pts  = np.asarray(pts, dtype=float)
        vel  = self._call_julia(pts)
        if blank_interior:
            try:
                vel[self.mesh.contains(pts)] = np.nan
            except Exception:
                pass
        return vel

    def generate_flow_field(self, x, y, z, zero_inside=True):
        """Same signature as VortexSheetSolver.generate_flow_field."""
        pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        vel = self._call_julia(pts)
        if zero_inside:
            try:
                vel[self.mesh.contains(pts)] = 0.0
            except Exception:
                pass
        return vel   # (N, 3)

    # stubs kept for compatibility
    def bc_residual(self): return float("nan")
    def net_source(self):  return float("nan")
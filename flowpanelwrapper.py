import os
import subprocess
import tempfile

import numpy as np
import trimesh


class FLOWPanelSolver:
    """
    Python wrapper around the Julia panel solver in FP.jl.

    Important convention:
        The mesh and all query points must already be in the same coordinate
        system before this class is constructed.

    This wrapper does NOT secretly scale query points.

    If your STL is in mm and CFD/query points are in m, do:

        mesh = trimesh.load_mesh(path)
        mesh.apply_scale(1 / 1000)
        solver = FLOWPanelSolver(mesh, V_inf)

    Do not pass unscaled mesh and expect this wrapper to fix units.
    """

    def __init__(
        self,
        mesh,
        V_inf,
        julia_script="FP.jl",
        julia_bin="julia",
        verbose=True,
    ):
        # Accept either a trimesh object or a path string.
        if isinstance(mesh, str):
            self.mesh = trimesh.load_mesh(mesh)
        else:
            self.mesh = mesh.copy()

        self.V_inf = np.asarray(V_inf, dtype=float)
        self._julia_bin = julia_bin
        self._julia_script = julia_script
        self._verbose = verbose

        self.diag = float(np.linalg.norm(self.mesh.extents))

        # Write mesh to temp STL so Julia can read it.
        self._tmpdir = tempfile.mkdtemp()
        self.stl_path = os.path.join(self._tmpdir, "body.stl")
        self.mesh.export(self.stl_path)

        if self._verbose:
            print("\nFLOWPanelSolver initialized")
            print("---------------------------")
            print(f"Julia binary:  {self._julia_bin}")
            print(f"Julia script:  {self._julia_script}")
            print(f"Temp STL path: {self.stl_path}")
            print(f"V_inf:         {self.V_inf}")
            print(f"Mesh bounds:\n{self.mesh.bounds}")
            print(f"Mesh extents:  {self.mesh.extents}")
            print(f"Mesh diag:     {self.diag}")

    def _call_julia(self, pts):
        """
        Write pts to npy, call Julia, read result npy.

        pts must be shape (N, 3) and in same units/coordinates as self.mesh.
        """
        pts = np.asarray(pts, dtype=np.float64)

        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"pts must have shape (N, 3), got {pts.shape}")

        if np.isnan(pts).any():
            raise ValueError("NaNs found in query points passed to Julia.")

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
                pts_file,
                out_file,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

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
                    f"Julia finished successfully, but output file was not created:\n"
                    f"{out_file}"
                )

            vel = np.load(out_file)

        vel = np.asarray(vel, dtype=float)

        if vel.shape != pts.shape:
            raise RuntimeError(
                f"Julia returned velocity shape {vel.shape}, expected {pts.shape}"
            )

        return vel

    def velocity(self, pts, blank_interior=True, blank_near=False, near_tol=None, **_):
        """
        Evaluate velocity at arbitrary points.

        pts:
            shape (N, 3)

        blank_interior:
            If True, replace velocities inside the mesh with NaN.

        blank_near:
            If True and near_tol is given, replace velocities near the surface
            with NaN using trimesh proximity if available.
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
        Evaluate velocity on a meshgrid.

        x, y, z:
            arrays from np.meshgrid with same shape.

        Returns:
            flattened velocity array of shape (N, 3).
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

"""
cfd_sampler.py
--------------
Builds a fast IDW (Inverse Distance Weighting) sampler from a CFD dataframe
backed by a cKDTree.  The returned callable accepts (N,3) query points and
returns a CFDFields dataclass.  Turbulence fields and the velocity-gradient
tensor are optional, only returned when the matching columns exist.
"""

from __future__ import annotations
import os
from dataclasses import dataclass

import numpy as np
import joblib
from scipy.spatial import cKDTree


@dataclass
class CFDFields:
    """Interpolated CFD fields at N query points."""
    velocity:        np.ndarray          # (N, 3)  [vx, vy, vz]
    pressure:        np.ndarray          # (N,)
    turb_kin_energy: np.ndarray | None = None  # (N,)    k         -- None if not in CFD data
    turb_visc:       np.ndarray | None = None  # (N,)    mu_t      -- None if not in CFD data
    vel_gradient:    np.ndarray | None = None  # (N, 9)  du_i/dx_j -- None if not in CFD data


def _df_hash(df) -> str:
    return str(df.shape) + str(df.columns.tolist())


def build_cfd_sampler(df, n_points: int = 8, sharpness: float = 2.0,
                      cache_path: str = None):
    """
    Build (or load from cache) a fast IDW sampler from a CFD dataframe.

    Required columns : x/y/z-coordinate, x/y/z-velocity, pressure
    Optional columns : turb-kinetic-energy, viscosity-turb
                       dx-velocity-dx, dy-velocity-dx, dz-velocity-dx,
                       dx-velocity-dy, dy-velocity-dy, dz-velocity-dy,
                       dx-velocity-dz, dy-velocity-dz, dz-velocity-dz

    Returns
    -------
    sample(points) -> CFDFields
    """
    coords    = df[['x-coordinate', 'y-coordinate', 'z-coordinate']].to_numpy(dtype=float)
    base_cols = ['x-velocity', 'y-velocity', 'z-velocity', 'pressure']
    turb_cols = ['turb-kinetic-energy', 'viscosity-turb']
    grad_cols = ['dx-velocity-dx', 'dy-velocity-dx', 'dz-velocity-dx',
                 'dx-velocity-dy', 'dy-velocity-dy', 'dz-velocity-dy',
                 'dx-velocity-dz', 'dy-velocity-dz', 'dz-velocity-dz']
    has_turb  = all(c in df.columns for c in turb_cols)
    has_grad  = all(c in df.columns for c in grad_cols)

    # Column layout in `fields`: [base | turb? | grad?]. Offsets are tracked
    # so sample() can slice the right block back out after interpolation.
    cols       = base_cols + (turb_cols if has_turb else []) + (grad_cols if has_grad else [])
    turb_start = len(base_cols)
    grad_start = len(base_cols) + (len(turb_cols) if has_turb else 0)
    fields     = df[cols].to_numpy(dtype=float)

    # Try cache
    tree = None
    if cache_path and os.path.exists(cache_path):
        print("Loading sampler from cache...")
        cache = joblib.load(cache_path)
        if cache['hash'] == _df_hash(df):
            tree   = cache['tree']
            fields = cache['fields']

    if tree is None:
        print("Building cKDTree...")
        tree = cKDTree(coords)
        if cache_path:
            joblib.dump({'hash': _df_hash(df), 'tree': tree, 'fields': fields}, cache_path)

    tree.query(coords[:1], k=n_points, workers=-1)  # warm up

    def sample(points) -> CFDFields:
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)

        dists, idx = tree.query(pts, k=n_points, workers=-1)
        if n_points == 1:
            dists, idx = dists[:, None], idx[:, None]

        exact = dists[:, 0] == 0.0
        w = 1.0 / np.where(dists == 0.0, 1.0, dists) ** sharpness
        w[exact]    = 0.0
        w[exact, 0] = 1.0
        w /= w.sum(axis=1, keepdims=True)

        out = np.einsum('qk,qkf->qf', w, fields[idx])
        return CFDFields(
            velocity        = out[:, 0:3],
            pressure        = out[:, 3],
            turb_kin_energy = out[:, turb_start]     if has_turb else None,
            turb_visc       = out[:, turb_start + 1] if has_turb else None,
            vel_gradient    = out[:, grad_start:grad_start + 9] if has_grad else None,  # (N, 9)
        )

    return sample


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    PKL_PATH = r"inputs/csv_with_everything.pkl"
    df = pd.read_pickle(PKL_PATH)

    sample = build_cfd_sampler(df, n_points=4, sharpness=2.0)

    test_pts = np.array([[0, 0, 0], [1, 2, 3]], dtype=float)
    f = sample(test_pts)
    print("velocity:\n", f.velocity)
    print("pressure:", f.pressure)
    print("turb_kin_energy:", f.turb_kin_energy)
    print("vel_gradient shape:", None if f.vel_gradient is None else f.vel_gradient.shape)
"""
momentum_worker.py
==================

Minimal module holding ONLY the BPA surface-force worker that runs in a child
process. It is deliberately tiny: when multiprocessing uses the "spawn" start
method (the only option on Windows/macOS), the child imports the module that
contains the target function. If that function lived in GPR.py, the child would
re-import jax, trimesh, sklearn, FLOWPanel, etc. on every phase - several seconds
of wasted startup. Keeping the worker here means the child only imports this file
plus momentum.momentum_open (and numpy), so spawn is cheap.

pyvista/VTK and open3d are imported ONLY inside surface_force_bpa's module, in
the child, so their process-global state never touches the parent where the
alive_progress bar lives.
"""

import numpy as np


def bpa_force_worker(region_points, midpoint, mom_vel, mom_p, bpa_L, q):
    """Runs in a CHILD process; returns the force vector via queue q."""
    try:
        from momentum.momentum_open import surface_force_bpa
        FORCE, _mesh, _pcd = surface_force_bpa(
            region_points, midpoint, mom_vel, mom_p, L=bpa_L)
        q.put(("ok", np.asarray(FORCE, float)))
    except Exception as e:
        q.put(("err", repr(e)))

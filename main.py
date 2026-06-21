"""
main.py  --  wind-force pipeline
"""

import os
import numpy as np
import pandas as pd
import trimesh

from flowpanelwrapper import FLOWPanelSolver
from cfd_sampler import build_cfd_sampler
from cylinder_geom import (
    calculate_wake_cylinder_parameters,
    generate_cylindrical_sampling_coordinates,
    generate_momentum_integration_mesh,
)
from fit_gprs import fit_gprs
from momentum import surface_force

# ── Settings ─────────────────────────────────────────────────────────────────

STL_PATH  = "input_stls/Aerospecial_building4.stl"
CFD_PATH  = "inputs/csv_with_everything.pkl"
V_INF     = np.array([0.0, 13.6, 0.0])
RHO       = 1.225

R_FACTOR, H_FACTOR, TILT_DEG = 2.4, 1.4, 10.0
N_SIDE, N_TOP                 = 324, 76
N_MOM                         = 100_000

# ── One-time setup  (keep alive to avoid reloading) ──────────────────────────

stl    = trimesh.load_mesh(STL_PATH); stl.apply_scale(1e-3)
solver = FLOWPanelSolver(stl, V_INF, julia_script="FP.jl", julia_bin="julia")
cfd    = build_cfd_sampler(pd.read_pickle(CFD_PATH), n_points=4, sharpness=2.0)

# ── Cylinder geometry ─────────────────────────────────────────────────────────

c_bot, c_top, radius = calculate_wake_cylinder_parameters(
    stl, R_FACTOR, H_FACTOR, V_INF, tilt_deg=TILT_DEG)

# ── Drone points  (stack more here for adaptive sampling) ────────────────────

drone_pts  = generate_cylindrical_sampling_coordinates(
    c_bot, c_top, radius, z_clearance=0.1, side_points=N_SIDE, top_points=N_TOP)
cfd_fields = cfd(drone_pts)

# ── Fit GPRs ──────────────────────────────────────────────────────────────────

gprs = fit_gprs(drone_pts, cfd_fields, solver)

# ── Momentum mesh + prior (cached) ───────────────────────────────────────────

mom_mesh  = generate_momentum_integration_mesh(
    c_bot, c_top, radius, total_points=N_MOM, cap_top=True)
mom_pts   = np.asarray(mom_mesh.points, dtype=float)

cache = f"cache/prior_{hash((STL_PATH,*V_INF,*c_bot,*c_top,round(radius,4),N_MOM))}.npy"
os.makedirs("cache", exist_ok=True)
prior_mom = np.load(cache) if os.path.exists(cache) else np.save(
    cache, solver.velocity(mom_pts, blank_interior=True)) or np.load(cache)

# ── Reconstruct + integrate force ────────────────────────────────────────────

mom_mesh["velocity"], mom_mesh["pressure"] = gprs.predict(mom_pts, prior_mom)

F = surface_force(mom_mesh, rho=RHO, interior_point=stl.centroid)
print(f"F = {F/1e3} kN   |F| = {np.linalg.norm(F):.4g} N")

solver.close()
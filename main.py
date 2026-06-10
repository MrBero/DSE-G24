"""
main.py
-------
Wind-field force pipeline. Wires together:
    cylinder_geom     -- sampling geometry
    cfd_sampler       -- IDW (Inverse Distance Weighting) interpolation of CFD data
    flowpanelwrapper  -- Julia FLOWPanel potential-flow prior
    divergence_free_gpr -- divergence-free vector GPR (Gaussian Process Regression) for velocity
    momentum          -- surface-force integration

Inputs (hard-coded paths, edit below if needed):
    inputs/csv_with_everything.pkl      -- CFD data
    input_stls/Aerospecial_building4.stl -- building geometry STL
"""

import numpy as np
import pandas as pd
import trimesh

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from cylinder_geom import (
	calculate_wake_cylinder_parameters,
	generate_cylindrical_sampling_coordinates,
	generate_momentum_integration_mesh,
)
from cfd_sampler import build_cfd_sampler
from flowpanelwrapper import FLOWPanelSolver
from divergence_free_gpr import DivergenceFreeGPR
from momentum import surface_force

# =============================================================================
# Configuration
# =============================================================================

STL_PATH        = r"input_stls/Aerospecial_building4.stl"
CFD_PKL_PATH    = r"inputs/csv_with_everything.pkl"
CACHE_PATH      = r"inputs/cfd_sampler_cache.joblib"

V_INF           = np.array([0.0, 12.6, 0.0])   # free-stream velocity (m/s)
RHO             = 1.225                          # air density (kg/m^3)
STL_SCALE       = 1.0 / 1000.0                  # STL is in mm, convert to m

# Cylinder geometry multipliers (passed to calculate_wake_cylinder_parameters)
R_FACTOR        = 1.5    # cylinder radius = R_FACTOR * building footprint circumradius
H_FACTOR        = 1.2    # cylinder height = H_FACTOR * building height
TILT_DEG        = 23.0   # downstream wake tilt in degrees

# Sampling
N_DRONE_POINTS  = 300    # points used to train the GPR
N_MOM_POINTS    = 25_000 # points on the momentum integration surface

# GPR settings
N_RESTARTS      = 8      # L-BFGS-B (Limited-memory Broyden-Fletcher-Goldfarb-Shanno Bounded) restarts for velocity hyperparameter optimisation
POSTERIOR_BATCH = 4_000  # chunk size to bound peak memory during prediction

# IDW (Inverse Distance Weighting) sampler settings
IDW_N_POINTS    = 4
IDW_SHARPNESS   = 2.0

# =============================================================================
# Helpers
# =============================================================================


def predict_batched(estimator, pts: np.ndarray, batch: int = 4_000) -> np.ndarray:
	"""Call estimator.predict in chunks to bound peak memory. Returns (N,)."""
	pts = np.asarray(pts, dtype=float)
	out = np.empty(pts.shape[0], dtype=float)
	for i in range(0, pts.shape[0], batch):
		out[i:i + batch] = estimator.predict(pts[i:i + batch])
	return out

# =============================================================================
# Main pipeline
# =============================================================================

def run():
	# ------------------------------------------------------------------
	# 1. Load geometry and start panel solver
	# ------------------------------------------------------------------
	print("Loading STL...")
	stl_mesh = trimesh.load_mesh(STL_PATH)
	stl_mesh.apply_scale(STL_SCALE)

	print("Starting Julia panel server...")
	solver = FLOWPanelSolver(stl_mesh, V_INF, julia_script="FP.jl",
	                         julia_bin="julia", verbose=True)

	# ------------------------------------------------------------------
	# 2. Determine cylinder geometry from the building mesh
	# ------------------------------------------------------------------
	print("Computing cylinder geometry...")
	center_bot, center_top, radius = calculate_wake_cylinder_parameters(
		stl_mesh, R_FACTOR, H_FACTOR, V_INF, tilt_deg=TILT_DEG
	)
	print(f"  bottom center : {center_bot}")
	print(f"  top center    : {center_top}")
	print(f"  radius        : {radius:.4f} m")

	# ------------------------------------------------------------------
	# 3. Generate sampling coordinates (drone points) and momentum mesh
	# ------------------------------------------------------------------
	print("Generating sampling coordinates...")
	drone_pts = generate_cylindrical_sampling_coordinates(
		center_bot, center_top, radius,
		total_points=N_DRONE_POINTS,
		cap_bottom=False, cap_top=True,
	)
	print(f"  drone points  : {drone_pts.shape[0]}")

	print("Generating momentum integration mesh...")
	mom_mesh = generate_momentum_integration_mesh(
		center_bot, center_top, radius,
		total_points=N_MOM_POINTS,
		cap_bottom=False, cap_top=True,
	)
	print(f"  momentum pts  : {mom_mesh.n_points}")

	# ------------------------------------------------------------------
	# 4. Sample CFD at drone locations
	# ------------------------------------------------------------------
	print("Loading CFD data and building IDW sampler...")
	df = pd.read_pickle(CFD_PKL_PATH)
	cfd_sample = build_cfd_sampler(
		df, n_points=IDW_N_POINTS, sharpness=IDW_SHARPNESS, cache_path=CACHE_PATH
	)

	cfd_at_drone = cfd_sample(drone_pts)          # (N_drone, 4): [vx,vy,vz,p]
	training_vels = cfd_at_drone[:, :3]           # (N_drone, 3)
	training_pres = cfd_at_drone[:, 3]            # (N_drone,)

	# ------------------------------------------------------------------
	# 5. Get panel-solver (potential flow) velocities at drone points
	# ------------------------------------------------------------------
	print("Evaluating potential-flow prior at drone points...")
	prior_at_drone = solver.velocity(drone_pts, blank_interior=False)  # (N_drone, 3)

	if np.isnan(prior_at_drone).any():
		raise RuntimeError("NaNs in potential-flow prior at drone points.")

	# ------------------------------------------------------------------
	# 6. Fit divergence-free velocity GPR on residuals
	# ------------------------------------------------------------------
	print("Fitting velocity GPR...")
	vel_residuals = training_vels - prior_at_drone   # (N_drone, 3)

	vel_gpr = DivergenceFreeGPR(
		n_restarts=N_RESTARTS,
		posterior_batch=POSTERIOR_BATCH,
	).fit(drone_pts, vel_residuals)

	print(f"  ell={vel_gpr.ell_}  var={vel_gpr.var_:.4g}  "
	      f"noise={vel_gpr.noise_:.4g}  nll={vel_gpr.nll_:.6g}")

	# ------------------------------------------------------------------
	# 7. Fit scalar pressure GPR (no prior subtracted -- panel solver
	#    produces no pressure, so raw CFD pressure is the target)
	# ------------------------------------------------------------------
	print("Fitting pressure GPR...")
	p_kernel = Matern(
		length_scale=[1.0, 1.0, 1.0],
		length_scale_bounds=(1e-2, 1e3),
		nu=2.5,
	)
	p_gpr = GaussianProcessRegressor(
		kernel=p_kernel, normalize_y=True,
		n_restarts_optimizer=8, random_state=0,
	).fit(drone_pts, training_pres)
	print(f"  fitted pressure kernel: {p_gpr.kernel_}")

	# ------------------------------------------------------------------
	# 8. Predict velocity + pressure at momentum mesh points,
	#    attach to mesh, compute surface force
	# ------------------------------------------------------------------
	print("Evaluating potential-flow prior at momentum mesh points...")
	mom_pts = np.asarray(mom_mesh.points, dtype=float)
	prior_at_mom = solver.velocity(mom_pts, blank_interior=True)   # (N_mom, 3)

	print("Predicting velocity posterior at momentum mesh points...")
	mom_vel = vel_gpr.predict(mom_pts, prior_at_mom)               # (N_mom, 3)

	print("Predicting pressure posterior at momentum mesh points...")
	mom_pres = predict_batched(p_gpr, mom_pts, batch=POSTERIOR_BATCH)  # (N_mom,)

	# Attach fields directly
	mom_mesh["velocity"] = mom_vel
	mom_mesh["pressure"] = mom_pres

	print("Computing surface force...")
	F = surface_force(mom_mesh, rho=RHO)

	F_mag = float(np.linalg.norm(F))
	print("\n=== Result ===")
	print(f"Force vector: Fx={F[0]/1000:.3f}e3  Fy={F[1]/1000:.3f}e3  Fz={F[2]/1000:.3f}e3 kN")
	print(f"  |F|          : {F_mag:.4g} N")

	# ------------------------------------------------------------------
	# Cleanup
	# ------------------------------------------------------------------
	solver.close()

	return {
		"force_vector" : F,
		"force_mag"    : F_mag,
		"vel_gpr"      : vel_gpr,
		"p_gpr"        : p_gpr,
	}


if __name__ == "__main__":
	result = run()
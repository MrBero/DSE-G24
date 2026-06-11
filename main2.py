"""
main.py
-------
Wind-field force pipeline. Wires together:
	cylinder_geom        -- sampling geometry
	cfd_sampler          -- IDW interpolation of CFD data
	flowpanelwrapper     -- Julia FLOWPanel potential-flow prior
	divergence_free_gpr  -- divergence-free vector GPR for velocity
	momentum             -- surface-force integration

Pipeline is split into distinct stages:
	setup()                 -- load STL, start panel solver, build CFD sampler
	build_cylinder_ctx()    -- cylinder geometry + momentum mesh
	generate_drone_points() -- geometric point generation
	fit_gpr_to_samples()    -- GPR fitting on drone samples
	evaluate_force()        -- momentum integration and surface force calculation
"""

import hashlib
import os
import time
import numpy as np
import pandas as pd
import trimesh

from dataclasses import dataclass, field, replace
from typing import Optional

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
from visualize_pipeline import visualize


# =============================================================================
# Configuration Domains
# =============================================================================

@dataclass(frozen=True)
class EnvConfig:
	"""Immutable physical and environmental setup."""
	stl_path:     str
	cfd_pkl_path: str
	v_inf:        np.ndarray
	stl_scale:    float


@dataclass(frozen=True)
class StudyConfig:
	"""Mutable algorithmic hyperparameters for a given pipeline run."""
	rho:          float
	radius:       float
	height:       float
	tilt_deg:     float
	n_side:       int
	n_top:        int
	depth:        float


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class SetupContext:
	stl_mesh:          trimesh.Trimesh
	solver:            FLOWPanelSolver
	cfd_sample:        object           
	building_centroid: np.ndarray     
	stl_path:          str          
	v_inf:             np.ndarray   


@dataclass
class CylinderContext:
	center_bot:   np.ndarray
	center_top:   np.ndarray
	radius:       float
	height:       float
	tilt_deg:     float
	mom_mesh:     object


@dataclass
class FitResult:
	vel_gpr:      DivergenceFreeGPR
	p_gpr:        GaussianProcessRegressor
	drone_pts:    np.ndarray          
	drone_vels:   np.ndarray          
	extra_tags:   dict = field(default_factory=dict)  


@dataclass
class ForceResult:
	force_vector: np.ndarray   
	force_mag:    float        
	fit:          FitResult
	cylinder_ctx: CylinderContext


# =============================================================================
# Helpers
# =============================================================================

def predict_batched(estimator, pts: np.ndarray, batch: int = 4000) -> np.ndarray:
	pts = np.asarray(pts, dtype=float)
	out = np.empty(pts.shape[0], dtype=float)
	for i in range(0, pts.shape[0], batch):
		out[i:i + batch] = estimator.predict(pts[i:i + batch])
	return out


def _fmt(seconds: float) -> str:
	if seconds < 60:
		return f"{seconds:.2f}s"
	m, s = divmod(seconds, 60)
	return f"{int(m)}m {s:.2f}s"


def _prior_mom_cache_path(
	stl_path, v_inf, center_bot, center_top, radius, n_mom_points, cap_bottom, cap_top
) -> str:
	bits = (
		stl_path, tuple(v_inf), tuple(center_bot), tuple(center_top),
		round(radius, 6), n_mom_points, cap_bottom, cap_top,
	)
	h = hashlib.md5(str(bits).encode()).hexdigest()[:10]
	return f"cache/prior_mom_{h}.npy"


# =============================================================================
# Stage 1 -- Setup
# =============================================================================

def setup(
	stl_path:      str,
	cfd_pkl_path:  str,
	v_inf:         np.ndarray,
	stl_scale:     float = 0.001,
	idw_n_points:  int = 4,
	idw_sharpness: float = 2.0,
) -> SetupContext:
	print("=" * 60)
	print("STAGE 1 -- Setup")
	print("=" * 60)

	print("Loading STL...")
	t0 = time.perf_counter()
	stl_mesh = trimesh.load_mesh(stl_path)
	stl_mesh.apply_scale(stl_scale)
	print(f"  [done in {_fmt(time.perf_counter() - t0)}]")

	print("Starting Julia panel server...")
	t0 = time.perf_counter()
	solver = FLOWPanelSolver(
		stl_mesh, v_inf,
		julia_script="FP.jl",
		julia_bin="julia",
		verbose=True,
	)
	print(f"  [done in {_fmt(time.perf_counter() - t0)}]")

	print("Loading CFD data and building IDW sampler...")
	t0 = time.perf_counter()
	df = pd.read_pickle(cfd_pkl_path)
	cfd_sample = build_cfd_sampler(df, n_points=idw_n_points, sharpness=idw_sharpness)
	print(f"  [done in {_fmt(time.perf_counter() - t0)}]")

	building_centroid = np.array(stl_mesh.centroid, dtype=float)

	return SetupContext(
		stl_mesh=stl_mesh,
		solver=solver,
		cfd_sample=cfd_sample,
		building_centroid=building_centroid,
		stl_path=stl_path,
		v_inf=v_inf,
	)


# =============================================================================
# Stage 2 -- Cylinder context
# =============================================================================

def build_cylinder_ctx(
	radius:       float,
	height:       float,
	center_bot:   np.ndarray,
	center_top:   np.ndarray,
	tilt_deg:     float,
	n_mom_points: int = 100_000,
) -> CylinderContext:
	"""Strictly accepts physical parameters. No internal calculations of centers."""
	print("=" * 60)
	print(f"STAGE 2 -- Cylinder context (Radius={radius:.2f}m, Height={height:.2f}m, tilt={tilt_deg}deg)")
	print("=" * 60)

	print("Generating momentum integration mesh...")
	t0 = time.perf_counter()
	mom_mesh = generate_momentum_integration_mesh(
		center_bot, center_top, radius,
		total_points=n_mom_points,
		cap_bottom=False,
		cap_top=True,
	)
	print(f"  momentum pts  : {mom_mesh.n_points}")
	print(f"  [done in {_fmt(time.perf_counter() - t0)}]")

	return CylinderContext(
		center_bot=center_bot,
		center_top=center_top,
		radius=radius,
		height=height,
		tilt_deg=tilt_deg,
		mom_mesh=mom_mesh,
	)


# =============================================================================
# Stage 3 -- Sample Generation and Fitting
# =============================================================================

def generate_drone_points(
	cyl: CylinderContext, 
	n_side: int, 
	n_top: int, 
	depth: float = 0.0
) -> np.ndarray:
	"""Pure geometry operation. No CFD dependencies."""
	print("Generating sampling coordinates...")
	t0 = time.perf_counter()
	
	if depth > 0:
		inner = generate_cylindrical_sampling_coordinates(
			cyl.center_bot, cyl.center_top, cyl.radius - depth,
			z_clearance=0.1, side_points=n_side // 2, top_points=n_top, bot_points=0,
		)
		outer = generate_cylindrical_sampling_coordinates(
			cyl.center_bot, cyl.center_top, cyl.radius + depth,
			z_clearance=0.1, side_points=n_side // 2, top_points=0, bot_points=0,
		)
		pts = np.vstack([inner, outer])
	else:
		pts = generate_cylindrical_sampling_coordinates(
			cyl.center_bot, cyl.center_top, cyl.radius,
			z_clearance=0.1, side_points=n_side, top_points=n_top, bot_points=0,
		)
		
	print(f"  total points  : {pts.shape[0]}")
	print(f"  [done in {_fmt(time.perf_counter() - t0)}]")
	return pts


def fit_gpr_to_samples(
	ctx:             SetupContext,
	drone_pts:       np.ndarray,
	n_restarts:      int = 8,
	posterior_batch: int = 4000,
	extra_tags:      dict = None,
) -> FitResult:
	"""Always takes explicit points. Extracts CFD data and fits models."""
	if extra_tags is None:
		extra_tags = {}

	print("=" * 60)
	print(f"STAGE 3 -- Fitting GPR ({drone_pts.shape[0]} points)")
	print("=" * 60)

	print("Sampling CFD at drone points...")
	t0 = time.perf_counter()
	cfd_at_drone  = ctx.cfd_sample(drone_pts)   
	training_vels = cfd_at_drone[:, :3]
	training_pres = cfd_at_drone[:, 3]
	print(f"  [done in {_fmt(time.perf_counter() - t0)}]")

	print("Evaluating potential-flow prior at drone points...")
	t0 = time.perf_counter()
	prior_at_drone = ctx.solver.velocity(drone_pts, blank_interior=False)
	if np.isnan(prior_at_drone).any():
		raise RuntimeError("NaNs in potential-flow prior at drone points.")
	print(f"  [done in {_fmt(time.perf_counter() - t0)}]")

	print("Fitting velocity GPR...")
	t0 = time.perf_counter()
	vel_residuals = training_vels - prior_at_drone
	vel_gpr = DivergenceFreeGPR(
		n_restarts=n_restarts,
		posterior_batch=posterior_batch,
	).fit(drone_pts, vel_residuals)
	print(f"  ell={vel_gpr.ell_}  var={vel_gpr.var_:.4g}  "
		  f"noise={vel_gpr.noise_:.4g}  nll={vel_gpr.nll_:.6g}")
	print(f"  [done in {_fmt(time.perf_counter() - t0)}]")

	print("Fitting pressure GPR...")
	t0 = time.perf_counter()
	p_kernel = Matern(
		length_scale=[1.0, 1.0, 1.0],
		length_scale_bounds=(1e-2, 1e3),
		nu=2.5,
	)
	p_gpr = GaussianProcessRegressor(
		kernel=p_kernel, normalize_y=True,
		n_restarts_optimizer=n_restarts, random_state=0,
	).fit(drone_pts, training_pres)
	print(f"  fitted pressure kernel: {p_gpr.kernel_}")
	print(f"  [done in {_fmt(time.perf_counter() - t0)}]")

	return FitResult(
		vel_gpr=vel_gpr,
		p_gpr=p_gpr,
		drone_pts=drone_pts,
		drone_vels=training_vels,
		extra_tags=extra_tags,
	)


# =============================================================================
# Stage 4 -- Evaluate Force
# =============================================================================

def evaluate_force(
	*, 
	setup_ctx:             SetupContext,
	cylinder_ctx:          CylinderContext,
	fit_result:            FitResult,
	air_density:           float,
	prediction_batch_size: int = 4000,
) -> ForceResult:
	"""
	Keyword-only arguments to explicitly map operational context. 
	Evaluates fitted GPRs on momentum mesh and integrates surface forces.
	"""
	print("=" * 60)
	print("STAGE 4 -- Evaluate force")
	print("=" * 60)

	mom_pts = np.asarray(cylinder_ctx.mom_mesh.points, dtype=float)

	cache_path = _prior_mom_cache_path(
		setup_ctx.stl_path, setup_ctx.v_inf,
		cylinder_ctx.center_bot, cylinder_ctx.center_top, cylinder_ctx.radius,
		mom_pts.shape[0], False, True,
	)
	
	print("Evaluating potential-flow prior at momentum mesh points...")
	t0 = time.perf_counter()
	if os.path.exists(cache_path):
		print(f"  Loading cached prior: {cache_path}")
		prior_at_mom = np.load(cache_path)
	else:
		prior_at_mom = setup_ctx.solver.velocity(mom_pts, blank_interior=True)
		os.makedirs("cache", exist_ok=True)
		np.save(cache_path, prior_at_mom)
		print(f"  Saved prior cache: {cache_path}")
	print(f"  [done in {_fmt(time.perf_counter() - t0)}]")

	print("Predicting velocity posterior at momentum mesh points...")
	t0 = time.perf_counter()
	mom_vel = fit_result.vel_gpr.predict(mom_pts, prior_at_mom)
	print(f"  [done in {_fmt(time.perf_counter() - t0)}]")

	print("Predicting pressure posterior at momentum mesh points...")
	t0 = time.perf_counter()
	mom_pres = predict_batched(fit_result.p_gpr, mom_pts, batch=prediction_batch_size)
	print(f"  [done in {_fmt(time.perf_counter() - t0)}]")

	cylinder_ctx.mom_mesh["velocity"] = mom_vel
	cylinder_ctx.mom_mesh["pressure"] = mom_pres

	print("Computing surface force...")
	t0 = time.perf_counter()
	F = surface_force(cylinder_ctx.mom_mesh, rho=air_density, interior_point=setup_ctx.building_centroid)
	print(f"  [done in {_fmt(time.perf_counter() - t0)}]")

	F_mag = float(np.linalg.norm(F))
	print(f"  Fx={F[0]/1000:.3f}  Fy={F[1]/1000:.3f}  Fz={F[2]/1000:.3f} kN  |F|={F_mag:.4g} N")

	return ForceResult(
		force_vector=F, 
		force_mag=F_mag, 
		fit=fit_result,
		cylinder_ctx=cylinder_ctx
	)


# =============================================================================
# Results helpers
# =============================================================================

def results_to_dataframe(results: list[ForceResult]) -> pd.DataFrame:
	rows = []
	for r in results:
		fit = r.fit
		cyl = r.cylinder_ctx
		row = {
			"Fx_N":        r.force_vector[0],
			"Fy_N":        r.force_vector[1],
			"Fz_N":        r.force_vector[2],
			"F_mag_N":     r.force_mag,
			"n_drones":    fit.drone_pts.shape[0],
			"radius":      cyl.radius,
			"height":      cyl.height,
			"tilt_deg":    cyl.tilt_deg,
		}
		row.update(fit.extra_tags)
		rows.append(row)
	return pd.DataFrame(rows)


# =============================================================================
# Main 
# =============================================================================

def main():
	total_start = time.perf_counter()

	# ------------------------------------------------------------------
	# 1. Define the immutable environment
	# ------------------------------------------------------------------
	env = EnvConfig(
		stl_path=r"input_stls/Aerospecial_building4.stl",
		cfd_pkl_path=r"inputs/csv_with_everything.pkl",
		v_inf=np.array([0.0, 13.6, 0.0]),
		stl_scale=0.001
	)

	# Stage 1: Setup
	ctx = setup(
		stl_path=env.stl_path,
		cfd_pkl_path=env.cfd_pkl_path,
		v_inf=env.v_inf,
		stl_scale=env.stl_scale,
	)

	# ------------------------------------------------------------------
	# Calculate baseline physical dimensions using factors (one time)
	# ------------------------------------------------------------------
	bot_base, top_base, radius_base = calculate_wake_cylinder_parameters(
		ctx.stl_mesh, r_factor=2.4, h_factor=1.4, v_inf=env.v_inf, tilt_deg=10.0
	)
	height_base = float(np.linalg.norm(top_base - bot_base))

	# ------------------------------------------------------------------
	# 2. Define the baseline hyperparameter configuration
	# ------------------------------------------------------------------
	base_study = StudyConfig(
		rho=1.225,
		radius=radius_base,
		height=height_base,
		tilt_deg=10.0,
		n_side=324,
		n_top=76,
		depth=0.0
	)
	
	results: list[ForceResult] = []

	# ------------------------------------------------------------------
	# Example A: convergence study over drone density
	# ------------------------------------------------------------------
	# Build the static geometry context for the density sweep
	cyl = build_cylinder_ctx(
		radius=base_study.radius,
		height=base_study.height,
		center_bot=bot_base,
		center_top=top_base,
		tilt_deg=base_study.tilt_deg,
	)

	for n in [100, 200, 324, 500]:
		print(f"\n--- Convergence run: n_drones_side={n} ---")
		
		run_config = replace(base_study, n_side=n)
		
		pts = generate_drone_points(
			cyl, 
			n_side=run_config.n_side, 
			n_top=run_config.n_top, 
			depth=run_config.depth
		)
		
		fit = fit_gpr_to_samples(
			ctx, drone_pts=pts,
			extra_tags={"study": "drone_density", "drone_depth": run_config.depth, "rho": run_config.rho}
		)
		
		result = evaluate_force(
			setup_ctx=ctx, 
			cylinder_ctx=cyl, 
			fit_result=fit,
			air_density=run_config.rho,
		)
		results.append(result)

	# ------------------------------------------------------------------
	# Example B: Absolute Radius sweep
	# ------------------------------------------------------------------
	# We sweep absolute radius directly, generating new physical centers each time
	for r_abs in [40.0, 45.0, 50.0, 55.0]:
		print(f"\n--- Radius sweep: Radius={r_abs}m ---")
		
		run_config = replace(base_study, radius=r_abs)
		
		# (Optional) If your helper calculates centers based on radius, use it here.
		# Assuming bot_base and top_base remain fixed for a pure radius sweep:
		cyl_r = build_cylinder_ctx(
			radius=run_config.radius,
			height=run_config.height,
			center_bot=bot_base,
			center_top=top_base,
			tilt_deg=run_config.tilt_deg,
		)
		
		pts = generate_drone_points(
			cyl_r, 
			n_side=run_config.n_side, 
			n_top=run_config.n_top, 
			depth=run_config.depth
		)
		
		fit = fit_gpr_to_samples(
			ctx, drone_pts=pts,
			extra_tags={"study": "radius_sweep", "rho": run_config.rho}
		)
		
		result = evaluate_force(
			setup_ctx=ctx, 
			cylinder_ctx=cyl_r, 
			fit_result=fit,
			air_density=run_config.rho,
		)
		results.append(result)

	df = results_to_dataframe(results)
	print("\n=== Results ===")
	print(df.to_string(index=False))
	df.to_csv("results.csv", index=False)
	print("\nSaved: results.csv")

	print(f"\n=== Total pipeline time: {_fmt(time.perf_counter() - total_start)} ===")

	last = results[-1]
	ctx.solver.close()

	visualize(
		mom_mesh=last.cylinder_ctx.mom_mesh,
		stl_mesh=ctx.stl_mesh,
		drone_pts=last.fit.drone_pts,
		drone_vels=last.fit.drone_vels,
		show=True,
	)

	return df


if __name__ == "__main__":
	main()
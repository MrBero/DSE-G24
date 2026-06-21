"""
momentum.py
-----------
Surface force integration via pressure and momentum flux on a pyvista mesh.

	F = ∮ ( -p n̂  -  ρ u(u·n̂)  -  ρ τ·n̂ ) dA

where τ = 2(mu_t/rho) S - (2/3) k I is the turbulent Reynolds-stress tensor
(Boussinesq hypothesis). The τ term is added automatically whenever the
mesh carries "turb_kin_energy", "turb_visc" and "vel_gradient" point data
alongside "velocity" and "pressure". It is only useful on control surfaces
passing through the wake, and is skipped entirely when those fields are not
on the mesh.
"""

from __future__ import annotations
import numpy as np
import pyvista as pv


def surface_force(
	mesh: pv.PolyData,
	rho: float = 1.225,
	interior_point: np.ndarray | None = None,
) -> np.ndarray:
	"""
	Net aerodynamic force on a closed surface mesh.

	Parameters
	----------
	mesh             : pv.PolyData with "velocity" (N,3) and "pressure" (N,)
					   point arrays. The turbulent Reynolds-stress term
					   (Boussinesq hypothesis) is added automatically when
					   the mesh also carries "turb_kin_energy", "turb_visc"
					   and "vel_gradient" (N,9) point arrays. If any one of
					   those three is missing, the Reynolds-stress term is
					   skipped rather than estimated some other way.
	rho              : fluid density  (kg/m³)
	interior_point   : point strictly inside the control volume; normals are
					   flipped to point outward from it.  Pass the building
					   centroid or cylinder axis midpoint.

	Returns
	-------
	F : (3,) force vector [Fx, Fy, Fz]  (N)
	"""
	required = ["velocity", "pressure"]
	for f in required:
		if f not in mesh.point_data:
			raise ValueError(f"Mesh missing point array '{f}'.")

	reynolds_fields     = ["turb_kin_energy", "turb_visc", "vel_gradient"]
	have_reynolds       = [f in mesh.point_data for f in reynolds_fields]
	include_reynolds    = all(have_reynolds)
	if any(have_reynolds) and not include_reynolds:
		missing = [f for f, present in zip(reynolds_fields, have_reynolds) if not present]
		print(f"  [surface_force] skipping Reynolds-stress term, missing {missing}")

	mesh = mesh.compute_normals(
		cell_normals=True, point_normals=False,
		consistent_normals=True,
		auto_orient_normals=(interior_point is None),
	).point_data_to_cell_data()

	n = mesh.cell_data["Normals"].copy()

	if interior_point is not None:
		interior_point = np.asarray(interior_point, dtype=float)
		centers = np.array(mesh.cell_centers().points)
		flip    = ((centers - interior_point) * n).sum(axis=1) < 0
		n[flip] *= -1
		if flip.any():
			print(f"  [surface_force] flipped {flip.sum()}/{len(flip)} normals outward")

	v   = mesh.cell_data["velocity"]
	p   = mesh.cell_data["pressure"]
	v_n = (v * n).sum(axis=1, keepdims=True)

	total = -p[:, None] * n - rho * v * v_n

	if include_reynolds:
		dudx  = mesh.cell_data["vel_gradient"].reshape(-1, 3, 3)
		mu_t  = mesh.cell_data["turb_visc"]
		k     = mesh.cell_data["turb_kin_energy"]
		S     = 0.5 * (dudx + dudx.transpose(0, 2, 1))
		tau   = 2.0 * (mu_t[:, None, None] / rho) * S - (2.0/3.0) * k[:, None, None] * np.eye(3)
		total -= rho * (tau @ n[:, :, None]).squeeze(-1)


	mesh.cell_data["total_force"] = total
	return mesh.integrate_data().cell_data["total_force"][0]


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
	import pandas as pd
	import trimesh
	from cfd_sampler import build_cfd_sampler
	from cylinder_geom import (
		calculate_wake_cylinder_parameters,
		generate_momentum_integration_mesh,
	)

	V_INF    = np.array([0.0, 13.6, 0.0])
	STL_PATH = r"input_stls/wind_turbine3_fixed.stl"
	PKL_PATH = r"inputs/csv_wind_turbine_5x_everything.pkl"

	df = pd.read_pickle(PKL_PATH)
	df.columns = df.columns.str.lower().str.replace(' ', '', regex=True)
	print(df.columns)
	sample = build_cfd_sampler(df, n_points=3, sharpness=3)

	stl = trimesh.load(STL_PATH)
	stl.apply_scale(0.001)

	bot, top, radius = calculate_wake_cylinder_parameters(stl, 1.4, 2, V_INF, 0)
	radius = 100.0   # override to 200 m diameter

	mesh = generate_momentum_integration_mesh(bot, top, radius, 100_000,
											  cap_bottom=False, cap_top=True)
	cfd = sample(mesh.points)
	mesh["velocity"] = cfd.velocity
	mesh["pressure"] = cfd.pressure
	if all(v is not None for v in [cfd.turb_kin_energy, cfd.turb_visc, cfd.vel_gradient]) :
		mesh["turb_kin_energy"] = cfd.turb_kin_energy
		mesh["turb_visc"]       = cfd.turb_visc
		mesh["vel_gradient"] = cfd.vel_gradient

	F = surface_force(mesh, rho=1.0)
	print("Force:", F)
import numpy as np
import pyvista as pv


def make_cylinder_mesh(
	center_xy: np.ndarray,
	z_bottom: float,
	z_top: float,
	radius: float,
	total_points: int = 100_000,
	top: bool = False,
	bottom: bool = False,
) -> pv.PolyData:
	height = z_top - z_bottom
	cx, cy = float(center_xy[0]), float(center_xy[1])
	cz = z_bottom + height / 2

	n_caps = int(top) + int(bottom)
	lateral_area = 2 * np.pi * radius * height
	cap_area      = np.pi * radius**2
	total_area    = lateral_area + n_caps * cap_area

	# Cell edge length s derived from total area and point count
	# total_points ~ total_area / s^2  =>  s = sqrt(total_area / total_points)
	s = np.sqrt(total_area / total_points)

	theta_res = max(4, round(2 * np.pi * radius / s))
	z_res     = max(1, round(height / s))
	# Disc: theta matches lateral; radial steps so dr ~ s
	cap_r_res = max(1, round(radius / s))

	mesh = pv.CylinderStructured(
		radius=radius,
		height=height,
		center=(cx, cy, cz),
		direction=(0, 0, 1),
		theta_resolution=theta_res,
		z_resolution=z_res,
	)
	surface = mesh.extract_surface(algorithm="dataset_surface")

	caps = []
	if bottom:
		caps.append(pv.Disc(
			center=(cx, cy, z_bottom),
			normal=(0, 0, -1),
			inner=0,
			outer=radius,
			r_res=cap_r_res,
			c_res=theta_res,
		))
	if top:
		caps.append(pv.Disc(
			center=(cx, cy, z_top),
			normal=(0, 0, 1),
			inner=0,
			outer=radius,
			r_res=cap_r_res,
			c_res=theta_res,
		))

	if caps:
		return surface.merge(caps)
	return surface



def attach_cfd_fields(
	mesh: pv.PolyData,
	sampler,
) -> pv.PolyData:
	"""Interpolate CFD (Computational Fluid Dynamics) velocity and pressure
	onto every point of the mesh and store them as point arrays in-place.

	Args:
		mesh:    PyVista PolyData surface mesh.
		sampler: Callable that maps (N, 3) points to (N, 4) array [vx, vy, vz, p].

	Returns:
		The same mesh with "velocity" (N, 3) and "pressure" (N,) point arrays attached.
	"""
	out = sampler(mesh.points)
	mesh["velocity"] = out[:, :3]
	mesh["pressure"] = out[:, 3]
	return mesh


def surface_force(
	mesh: pv.PolyData,
	rho: float = 1.225,
) -> tuple[np.ndarray, pv.PolyData]:
	"""Compute the net aerodynamic force on a surface mesh.

	Expects "velocity" (N, 3) and "pressure" (N,) to already be attached
	as point data arrays on the mesh before calling.

	Args:
		mesh: PyVista PolyData with "velocity" and "pressure" point arrays.
		rho:  Fluid density in kg/m^3 (default 1.225).

	Returns:
		F:    (3,) net force vector [Fx, Fy, Fz].
		mesh: Mesh annotated with per-cell force arrays.
	"""
	if "velocity" not in mesh.point_data or "pressure" not in mesh.point_data:
		raise ValueError(
			"Mesh must have 'velocity' and 'pressure' point arrays. "
			"Call attach_cfd_fields(mesh, sampler) first."
		)

	mesh = mesh.compute_normals(
		cell_normals=True,
		point_normals=False,
		consistent_normals=True,
		auto_orient_normals=True,
	)

	mesh = mesh.point_data_to_cell_data()

	n = mesh.cell_data["Normals"]
	v = mesh.cell_data["velocity"]
	p = mesh.cell_data["pressure"]

	v_dot_n = (v * n).sum(axis=1, keepdims=True)

	mesh.cell_data["pressure_force"] = -p[:, np.newaxis] * n
	mesh.cell_data["momentum_force"] = -rho * v * v_dot_n
	mesh.cell_data["total_force"]    = mesh.cell_data["pressure_force"] + mesh.cell_data["momentum_force"]

	F = mesh.integrate_data().cell_data["total_force"][0]

	return F, mesh


if __name__ == "__main__":
	Radius = 52.6
	r_factor = 0.7

	R = Radius * r_factor
	H = 70.0
	N_POINTS = 4
	SHARPNESS = 2

	MIDPOINT = np.array([
		6524.591 / 1000,
		1.213 * 10**5 / 10**3,
		52.6 / 2
	])

	PKL_PATH = r"INTERP\csv_with_everything.pkl"
	CACHE_PATH = r"inputs/cfd_sampler_cache.joblib"

	z_res = 500
	theta_res = 500
	cap_r_res = 100

	import time
	import pandas as pd
	import matplotlib.pyplot as plt
	from INTERP import interpolation

	t = time.time()
	print('starting')

	mesh = make_cylinder_mesh(
		center_xy=np.array([MIDPOINT[0], MIDPOINT[1]]),
		z_bottom=0.0,
		z_top=H,
		radius=R,
		total_points=320,
		top=True,
		bottom=False,
	)
	print(time.time() - t, "mesh done, starting interpolation")
	t = time.time()

	df = pd.read_pickle(PKL_PATH)
	sampler = interpolation.build_cfd_sampler(df, n_points=N_POINTS, sharpness=SHARPNESS, cache_path=CACHE_PATH)
	print(time.time() - t, "sampler done")
	t = time.time()

	surface = attach_cfd_fields(mesh, sampler)
	print(time.time() - t, "interpolation done")
	t = time.time()

	F, result = surface_force(surface, rho=1.225)
	print(time.time() - t, "force done")
	print(f"Force vector : {F}")

	pl = pv.Plotter()
	pl.add_mesh(result, scalars="pressure", cmap="viridis", show_edges=True, opacity=0.8)
	pl.add_axes()
	pl.show()
	F_REF_Y = 208647.0

	# ---- convergence study: POINT --------------------------------
	# F_REF_Y = 208647.0
	# point_counts = np.unique(np.round(np.logspace(np.log10(320), np.log10(1_000_000), 20)).astype(int))
	# fy_values = []

	# for n in point_counts:
	# 	m = make_cylinder_mesh(
	# 		center_xy=np.array([MIDPOINT[0], MIDPOINT[1]]),
	# 		z_bottom=0.0, z_top=H, radius=R,
	# 		total_points=int(n), top=True, bottom=False,
	# 	)
	# 	attach_cfd_fields(m, sampler)
	# 	F_n, _ = surface_force(m, rho=1.225)
	# 	fy_values.append(F_n[1])
	# 	print(f"n={n:>9,}   Fy={F_n[1]:>12.1f}")

	# plt.figure()
	# plt.axhline(F_REF_Y, color="red", linestyle="--", label=f"Reference Fy = {F_REF_Y:,.0f} N")
	# plt.plot(point_counts, fy_values, marker="o")
	# plt.xscale("log")
	# plt.xlabel("Total mesh points")
	# plt.ylabel("Fy [N]")
	# plt.title("Convergence study - Fy vs mesh points")
	# plt.legend()
	# plt.tight_layout()
	# plt.show()

	# ---- convergence study: RADIUS ----------------------------------------
	r_factors = np.linspace(0.5, 2, 40)
	fy_r = []

	for rf in r_factors:
		R_rf = Radius * rf
		m = make_cylinder_mesh(
			center_xy=np.array([MIDPOINT[0], MIDPOINT[1]]),
			z_bottom=0.0, z_top=H, radius=R_rf,
			total_points=250_000, top=True, bottom=False,
		)
		attach_cfd_fields(m, sampler)
		F_n, _ = surface_force(m, rho=1.225)
		fy_r.append(F_n[1])
		print(f"r_factor={rf:.3f}   R={R_rf:.2f}   Fy={F_n[1]:>12.1f}")

	plt.figure()
	plt.axhline(F_REF_Y, color="red", linestyle="--", label=f"Reference Fy = {F_REF_Y:,.0f} N")
	plt.plot(r_factors, fy_r, marker="o")
	plt.xlabel("r_factor")
	plt.ylabel("Fy [N]")
	plt.title("Convergence study - Fy vs r_factor")
	plt.legend()
	plt.tight_layout()
	plt.show()
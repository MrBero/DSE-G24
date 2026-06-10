import pyvista as pv
import numpy as np


# def attach_cfd_fields(
# 	mesh: pv.PolyData,
# 	velocity: np.ndarray | None = None,
# 	pressure: np.ndarray | None = None,
# 	predictor=None,
# ) -> pv.PolyData:
# 	"""
# 	Attach velocity and pressure to the mesh, either from pre-computed
# 	arrays or by calling a predictor(pts) -> (N,4) callable.
# 	"""
# 	if predictor is not None:
# 		out = predictor(mesh.points)
# 		velocity = out[:, :3]
# 		pressure = out[:, 3]

# 	if velocity is None or pressure is None:
# 		raise ValueError("Provide either (velocity, pressure) arrays or a predictor callable.")

# 	mesh["velocity"] = velocity
# 	mesh["pressure"] = pressure
# 	return mesh


def surface_force(
	mesh: pv.PolyData,
	rho: float = 1.225,
	interior_point: np.ndarray | None = None,
) -> np.ndarray:
	"""
	Compute the net aerodynamic force on a surface mesh via pressure and
	momentum flux integration.

	Parameters
	----------
	mesh           : pv.PolyData  surface mesh with "velocity" (N,3) and "pressure" (N,) point arrays
	rho            : float        fluid density (default 1.225 kg/m³)
	interior_point : (3,) array   a point known to be strictly inside the control volume.
								  If provided, normals are oriented to point AWAY from it
								  (i.e. outward), bypassing PyVista's centroid heuristic.
								  Pass the building centroid or cylinder axis midpoint here.

	Returns
	-------
	F : (3,) net force vector [Fx, Fy, Fz]
	"""
	for field in ("velocity", "pressure"):
		if field not in mesh.point_data:
			raise ValueError(f"Mesh is missing point array '{field}'.")

	# Compute consistent cell normals — but do NOT auto-orient yet if we
	# have a reliable interior point; we'll handle orientation ourselves.
	mesh = mesh.compute_normals(
		cell_normals=True, point_normals=False,
		consistent_normals=True,
		auto_orient_normals=(interior_point is None),  # only use heuristic as fallback
	).point_data_to_cell_data()

	n = mesh.cell_data["Normals"].copy()

	if interior_point is not None:
		interior_point = np.asarray(interior_point, dtype=float)
		# Vector from interior point to each cell center
		centers = np.array(mesh.cell_centers().points)
		to_center = centers - interior_point  # (N_cells, 3)
		# Dot with current normal: positive means already pointing outward
		dots = (n * to_center).sum(axis=1)   # (N_cells,)
		# Flip any normals pointing inward
		flip = dots < 0
		n[flip] *= -1
		if flip.any():
			print(f"  [surface_force] flipped {flip.sum()} / {len(flip)} cell normals "
				  f"to point outward from interior point.")

	v = mesh.cell_data["velocity"]
	p = mesh.cell_data["pressure"]

	v_n = (v * n).sum(axis=1, keepdims=True)

	mesh.cell_data["Normals"]         = n   # write back so integrate_data uses corrected normals
	mesh.cell_data["pressure_force"]  = -p[:, None] * n
	mesh.cell_data["momentum_force"]  = -rho * v * v_n
	mesh.cell_data["total_force"]     =  mesh.cell_data["pressure_force"] + mesh.cell_data["momentum_force"]

	return mesh.integrate_data().cell_data["total_force"][0]


if __name__ == "__main__":
	import trimesh
	import pandas as pd
	from cylinder_geom import calculate_wake_cylinder_parameters, generate_momentum_integration_mesh
	from cfd_sampler import build_cfd_sampler

	STL_PATH  = r"input_stls/Aerospecial_building4.stl"
	V_INF     = np.array([0.0, 13.6, 0.0])
	R_FACTOR  = 2    # cylinder radius  = R_FACTOR  × building footprint circumradius
	H_FACTOR  = 1.4  # cylinder height  = H_FACTOR  × building height
	TILT_DEG  = 20   # downstream wake tilt (degrees)

	stl_mesh = trimesh.load_mesh(STL_PATH)
	stl_mesh.apply_scale(1e-3)

	bot, top, radius = calculate_wake_cylinder_parameters(
		stl_mesh, R_FACTOR, H_FACTOR, V_INF, tilt_deg=TILT_DEG
	)
	print(f"bottom center : {bot}\ntop center    : {top}\nradius        : {radius}")

	mesh = generate_momentum_integration_mesh(bot, top, radius, total_points=100_000, cap_top=True)

	df         = pd.read_pickle(r"inputs/csv_with_everything.pkl")
	cfd_sample = build_cfd_sampler(df, n_points=4, sharpness=2.0)
	cfd_fields = cfd_sample(mesh.points)

	mesh["velocity"] = cfd_fields[:, :3]
	mesh["pressure"] = cfd_fields[:,  3]

	building_centroid = np.array([
	stl_mesh.centroid[0],
	stl_mesh.centroid[1],
	stl_mesh.centroid[2],
])
	print(f"building centroid: {building_centroid}")


	F = surface_force(mesh, rho=1.225, interior_point=building_centroid)
	print("Force [N]:", F)
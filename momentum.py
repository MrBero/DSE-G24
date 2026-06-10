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


def compute_forces(
	mesh: pv.PolyData,
	rho: float,
) -> tuple[np.ndarray, pv.PolyData]:
	"""
	Integrate pressure and momentum forces over a triangulated surface.

	Expects "velocity" (N,3) and "pressure" (N,) to already be attached
	as point data arrays on the mesh (e.g. via attach_cfd_fields).

	Returns total force as a (3,) array and the annotated mesh.
	"""
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

	return F


def surface_force(
	mesh: pv.PolyData,
	rho: float = 1.225,
) -> tuple[np.ndarray, pv.PolyData]:
	"""
	Compute the net aerodynamic force on a surface mesh.

	Expects "velocity" and "pressure" point arrays to be present on the
	mesh before calling (attach them with attach_cfd_fields beforehand).

	Parameters
	----------
	mesh : pv.PolyData  surface mesh with "velocity" and "pressure" point arrays
	rho  : float        fluid density (default 1.225 kg/m^3)

	Returns
	-------
	F    : (3,) net force vector [Fx, Fy, Fz]
	mesh : pv.PolyData  mesh annotated with per-cell force arrays
	"""
	if "velocity" not in mesh.point_data or "pressure" not in mesh.point_data:
		raise ValueError(
			"Mesh must have 'velocity' and 'pressure' point arrays. "
			"attach the cfd fields beforehand."
		)

	return compute_forces(mesh, rho)



from cylinder_geom import calculate_wake_cylinder_parameters, generate_cylindrical_sampling_coordinates, generate_momentum_integration_mesh, visualize_points
if __name__ == "__main__":
	STL_PATH        = r"input_stls/Aerospecial_building4.stl"
	V_INF           = np.array([0.0, 13.6, 0.0])   
	R_FACTOR        = 3    # cylinder radius = R_FACTOR * building footprint circumradius
	H_FACTOR        = 1.4    # cylinder height = H_FACTOR * building height
	TILT_DEG        = 20   # downstream wake tilt in degrees
	import trimesh
	stl_mesh = trimesh.load_mesh(STL_PATH)
	stl_mesh.apply_scale(1.0 / 1000.0)
	bot,top,radius = calculate_wake_cylinder_parameters(
		stl_mesh, R_FACTOR, H_FACTOR, V_INF, tilt_deg=TILT_DEG
	)

	print("bottom center : ", bot)
	print("top center    : ", top)
	print("radius        : ", radius)

	mesh = generate_momentum_integration_mesh(
	bot, top, radius,
	total_points=100_000,
	cap_top=True,
	)
	from cfd_sampler import build_cfd_sampler
	import pandas as pd
	df = pd.read_pickle(r"inputs/csv_with_everything.pkl")
	cfd_sample = build_cfd_sampler(
		df, n_points=4, sharpness=2.0, #cache_path=CACHE_PATH
	)
	cfd_at_drone = cfd_sample(mesh.points)
	training_vels = cfd_at_drone[:, :3]
	training_pres = cfd_at_drone[:, 3]

	mesh['velocity'] = training_vels
	mesh['pressure'] = training_pres

	F =  surface_force(mesh, rho=1.225)
	print(F)
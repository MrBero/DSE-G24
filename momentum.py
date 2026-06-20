from __future__ import annotations
import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay


def _reynolds_stress_from_boussinesq(
	mesh: pv.PolyData,
	mu_t: np.ndarray,
	k: np.ndarray,
	rho: float,
) -> np.ndarray:
	"""
	Reconstruct the Reynolds stress tensor field u_i'u_j' (per cell) from an
	eddy-viscosity RANS solution via the Boussinesq hypothesis:

		u_i'u_j' = (mu_t / rho) * (dui/dxj + duj/dxi) - (2/3) k delta_ij

	Velocity gradients are computed numerically on the mesh point data
	(since mu_t / k are only available as interpolated point values, not
	analytic gradients), then mapped to cell data to match "Normals".

	Parameters
	----------
	mesh : pv.PolyData  must already have "velocity" point data
	mu_t : (N_points,)  turbulent viscosity, point data
	k    : (N_points,)  turbulent kinetic energy, point data
	rho  : float        fluid density

	Returns
	-------
	uiuj : (N_cells, 3, 3)  Reynolds stress tensor per cell
	"""
	grad = mesh.compute_derivative(scalars="velocity", gradient=True)
	# "gradient" point array is (N, 9), row-major: [du/dx, du/dy, du/dz, dv/dx, ...]
	dudx = grad.point_data["gradient"].reshape(-1, 3, 3)  # [point, d(u_i)/d(x_j)]

	# map point data -> cell data so it lines up with Normals / pressure / velocity
	grad_mesh = pv.PolyData(mesh.points, mesh.faces)
	grad_mesh.point_data["dudx"] = dudx.reshape(-1, 9)
	grad_mesh.point_data["mu_t"] = mu_t
	grad_mesh.point_data["k"]    = k
	grad_cell = grad_mesh.point_data_to_cell_data()

	dudx_c = grad_cell.cell_data["dudx"].reshape(-1, 3, 3)
	mu_t_c = grad_cell.cell_data["mu_t"]
	k_c    = grad_cell.cell_data["k"]

	S = 0.5 * (dudx_c + np.transpose(dudx_c, (0, 2, 1)))  # symmetric strain rate, per cell

	eye = np.eye(3)[None, :, :]
	uiuj = (2.0 * mu_t_c[:, None, None] / rho) * S - (2.0 / 3.0) * k_c[:, None, None] * eye

	return uiuj


def surface_force(
	mesh: pv.PolyData,
	rho: float = 1.225,
	interior_point: np.ndarray | None = None,
	include_reynolds_stress: bool = False,
) -> np.ndarray:
	"""
	Compute the net aerodynamic force on a surface mesh via pressure and
	momentum flux integration.

	Parameters
	----------
	mesh           : pv.PolyData  surface mesh with "velocity" (N,3) and "pressure" (N,) point arrays.
								  If include_reynolds_stress=True, also needs
								  "turb_kin_energy" (N,) and "turb_visc" (N,) point arrays.
	rho            : float        fluid density (default 1.225 kg/m³)
	interior_point : (3,) array   a point known to be strictly inside the control volume.
								  If provided, normals are oriented to point AWAY from it
								  (i.e. outward), bypassing PyVista's centroid heuristic.
								  Pass the building centroid or cylinder axis midpoint here.
	include_reynolds_stress : bool
								  If True, adds the turbulent momentum flux term
								  -rho * (u_i'u_j') . n to the force integral, reconstructed
								  from turb_visc / turb_kin_energy via the Boussinesq
								  hypothesis. Only meaningful on control surfaces that pass
								  through the flow (e.g. a wake plane) -- on the body wall
								  itself this term is ~0 by no-slip.

	Returns
	-------
	F : (3,) net force vector [Fx, Fy, Fz]
	"""
	required = ["velocity", "pressure"]
	if include_reynolds_stress:
		required += ["turb_kin_energy", "turb_visc"]
	for field in required:
		if field not in mesh.point_data:
			raise ValueError(f"Mesh is missing point array '{field}'.")

	# point_data_to_cell_data() drops the original point arrays, so grab
	# whatever we still need from point data (e.g. for the Reynolds stress
	# gradient computation, which needs point data, not cell data) before
	# the mesh gets reassigned below.
	if include_reynolds_stress:
		velocity_points = mesh.point_data["velocity"].copy()
		turb_kin_energy_points = mesh.point_data["turb_kin_energy"].copy()
		turb_visc_points = mesh.point_data["turb_visc"].copy()
		points_xyz = mesh.points.copy()
		faces = mesh.faces.copy()

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

	if include_reynolds_stress:
		grad_source = pv.PolyData(points_xyz, faces)
		grad_source.point_data["velocity"] = velocity_points

		uiuj = _reynolds_stress_from_boussinesq(
			grad_source,
			mu_t=turb_visc_points,
			k=turb_kin_energy_points,
			rho=rho,
		)
		# -rho * (u_i'u_j') . n_j, per cell
		reynolds_force = -rho * np.einsum('cij,cj->ci', uiuj, n)
		mesh.cell_data["reynolds_force"] = reynolds_force
		mesh.cell_data["total_force"]    = mesh.cell_data["total_force"] + reynolds_force

	return mesh.integrate_data().cell_data["total_force"][0]


if __name__ == "__main__":
	import numpy as np
	import pandas as pd
	import pyvista as pv
	from cfd_sampler import build_cfd_sampler
	from cylinder_geom import generate_momentum_integration_mesh, generate_cylindrical_sampling_coordinates, calculate_wake_cylinder_parameters
	import trimesh
	# Ensure these are imported from your respective modules:
	# from your_module import generate_box_momentum_integration_mesh
	# from your_module import surface_force

	# -------------------------------------------------------------------------
	# Config
	# -------------------------------------------------------------------------
	V_INF = np.array([0.0, 13.6, 0.0])
	STL_PATH = r'input_stls\wind_turbine3_fixed.stl'
	PKL_PATH = r'inputs\csv_wind_turbine_5x_everything.pkl'

	R_factor = 1.4
	h_factor = 2
	tilt_deg = 0


	
	df = pd.read_pickle(PKL_PATH)
	df.columns = df.columns.str.lower().str.replace(' ', '', regex=True)

	cfd_sample = build_cfd_sampler(df, n_points=3, sharpness=3)

	stl_mesh = trimesh.load(STL_PATH)
	stl_mesh.apply_scale(.001)


	bot, top, radius = calculate_wake_cylinder_parameters(stl_mesh,R_factor, h_factor, V_INF, tilt_deg)

	print(bot, top, radius)

	radius = 200/2

	mesh = generate_momentum_integration_mesh(bot, top, radius, 100_000, cap_bottom = False, cap_top = True)

	cfd_fields = cfd_sample(mesh.points)

	mesh['velocity']         = cfd_fields.velocity
	mesh['pressure']         = cfd_fields.pressure
	mesh['turb_kin_energy']  = cfd_fields.turb_kin_energy
	mesh['turb_visc']        = cfd_fields.turb_visc

	F = surface_force(mesh, rho=1, include_reynolds_stress=True)

	print(F)
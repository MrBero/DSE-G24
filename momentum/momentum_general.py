import pyvista as pv
import numpy as np


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

	return F, mesh


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
			"Call attach_cfd_fields(mesh, sampler) first."
		)

	return compute_forces(mesh, rho)


if __name__ == "__main__":
	from main import make_cylinder_mesh, attach_cfd_fields

	R, H = 3.0, 10.0

	mesh = make_cylinder_mesh(
		center_xy=np.array([0.0, 0.0]),
		z_bottom=0.0,
		z_top=H,
		radius=R,
		theta_res=120,
		z_res=120,
		cap_r_res=30,
		top=True,
		bottom=True,
	)

	# Uniform flow for analytical validation: F should be ~zero for closed surface
	num_points = mesh.n_points
	mesh["velocity"] = np.tile([10.0, 0.0, 0.0], (num_points, 1))
	mesh["pressure"] = np.full(num_points, 20.0)

	F, result = surface_force(mesh, rho=1.225)

	print(f"Force vector : {F}")
	print(f"|F|          : {np.linalg.norm(F):.6e}")
	print(f"Expected vol : {np.pi * R**2 * H:.4f}")

	pl = pv.Plotter()
	pl.add_mesh(result, scalars="pressure", cmap="viridis", show_edges=True, opacity=0.8)
	result.set_active_vectors("velocity")
	pl.add_axes()
	pl.show()
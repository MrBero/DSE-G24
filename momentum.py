import pyvista as pv
import numpy as np


def attach_cfd_fields(mesh: pv.PolyData, predictor) -> pv.PolyData:
	"""
	attach the velocities and pressures to the momentum mesh (the GPR predictor or if testing the CFD sampler itself)
	"""
	out = predictor(mesh.points)
	mesh["velocity"] = out[:, :3]
	mesh["pressure"] = out[:, 3]
	return mesh

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


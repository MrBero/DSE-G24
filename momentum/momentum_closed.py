import pyvista as pv
import numpy as np


def reconstruct_surface(points: np.ndarray, center: np.ndarray) -> tuple[pv.PolyData, np.ndarray]:
	"""
	Reconstruct a closed triangulated surface from a point cloud via
	spherical projection + convex hull (Delaunay 3D + surface extraction).

	Returns the mesh and the index array mapping mesh vertices back to
	the original points array.
	"""
	shifted   = points - center
	spherical = shifted / np.linalg.norm(shifted, axis=1, keepdims=True)

	cloud = pv.PolyData(spherical)
	cloud.point_data["orig_idx"] = np.arange(len(points))

	mesh     = cloud.delaunay_3d().extract_surface()
	idx      = mesh.point_data["orig_idx"]
	mesh.points = points[idx]

	return mesh, idx


def compute_forces(
	mesh: pv.PolyData,
	velocity: np.ndarray,
	pressure: float,
	rho: float,
) -> np.ndarray:
	"""
	Integrate pressure and momentum forces over a triangulated surface.

	Assumes uniform velocity and pressure across all cells.
	Returns total force as a (3,) array.
	"""
	mesh = mesh.compute_normals(
		cell_normals=True,
		point_normals=False,
		consistent_normals=True,
		auto_orient_normals=True,
	)

	n = mesh.cell_data["Normals"]
	v = np.broadcast_to(velocity, n.shape).copy()

	v_dot_n = (v * n).sum(axis=1, keepdims=True)

	mesh.cell_data["pressure_force"] = pressure * n
	mesh.cell_data["momentum_force"] = rho * v * v_dot_n
	mesh.cell_data["total_force"]    = mesh.cell_data["pressure_force"] + mesh.cell_data["momentum_force"]

	return mesh.integrate_data().cell_data["total_force"][0]


def surface_force(
	points: np.ndarray,
	center: np.ndarray,
	velocity: np.ndarray,
	pressure: float,
	rho: float = 1.225,
) -> np.ndarray:
	"""
	Compute the net aerodynamic force on a surface defined by a point cloud.

	Parameters
	----------
	points   : (N, 3) surface point cloud
	center   : (3,)  interior point used for surface reconstruction
	velocity : (3,)  free-stream velocity vector
	pressure : float uniform static pressure
	rho      : float fluid density (default 1.225 kg/m^3)

	Returns
	-------
	F : (3,) net force vector [Fx, Fy, Fz]
	"""
	points   = np.asarray(points,   dtype=float)
	center   = np.asarray(center,   dtype=float)
	velocity = np.asarray(velocity, dtype=float)

	mesh, _ = reconstruct_surface(points, center)
	return compute_forces(mesh, velocity, pressure, rho)


if __name__ == "__main__":
	def sample_cylinder(R, H, n, top_cap=True, bottom_cap=True, rng=None):
		if rng is None:
			rng = np.random.default_rng()
 
		area_lat = 2 * np.pi * R * H
		area_cap = np.pi * R**2
		area_top = area_cap if top_cap    else 0.0
		area_bot = area_cap if bottom_cap else 0.0
		total    = area_lat + area_top + area_bot
 
		n_top = int(n * area_top / total) if top_cap    else 0
		n_bot = int(n * area_bot / total) if bottom_cap else 0
		n_lat = n - n_top - n_bot
 
		parts = []
 
		if n_lat > 0:
			theta = rng.uniform(0, 2 * np.pi, n_lat)
			z     = rng.uniform(0, H, n_lat)
			parts.append(np.c_[R * np.cos(theta), R * np.sin(theta), z])
 
		for n_cap, z_val in [(n_top, H), (n_bot, 0.0)]:
			if n_cap > 0:
				r     = R * np.sqrt(rng.uniform(0, 1, n_cap))
				theta = rng.uniform(0, 2 * np.pi, n_cap)
				parts.append(np.c_[r * np.cos(theta), r * np.sin(theta), np.full(n_cap, z_val)])
 
		return np.vstack(parts)
 
	R, H = 3.0, 10.0
 
	points = sample_cylinder(R, H, n=80, rng=np.random.default_rng(67))
	center = np.array([0.0, 0.0, H / 2.0])
 
	F = surface_force(
		points=points,
		center=center,
		velocity=np.array([10.0, 0.0, 0.0]),
		pressure=20.0,
		rho=1.225,
	)
 
	print(f"Force vector : {F}")
	print(f"|F|          : {np.linalg.norm(F):.6e}")
	print(f"Expected vol : {np.pi * R**2 * H:.4f}")

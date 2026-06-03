import pyvista as pv
import numpy as np

# ============================================================
# Config
# ============================================================

R         = 3.0
H         = 10.0
RHO       = 1.225
N_POINTS  = 80
VELOCITY  = np.array([10.0, 0.0, 0.0])
PRESSURE  = 20.0
SEED      = 67

# ============================================================
# Helpers
# ============================================================

def sample_cylinder_surface(R, H, n, top_cap=False, bottom_cap=False, rng=None):
	"""Return n points uniformly distributed on a cylinder surface."""
	if rng is None:
		rng = np.random.default_rng()

	area_lat = 2 * np.pi * R * H
	area_cap = np.pi * R**2
	area_top = area_cap if top_cap  else 0.0
	area_bot = area_cap if bottom_cap else 0.0
	total    = area_lat + area_top + area_bot

	n_lat = n - int(n * area_top / total) - int(n * area_bot / total)
	n_top = int(n * area_top / total) if top_cap    else 0
	n_bot = int(n * area_bot / total) if bottom_cap else 0

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


def reconstruct_surface(points, center):
	"""
	Reconstruct a closed triangulated surface from a point cloud via
	spherical projection + convex hull (Delaunay 3D + surface extraction).
	"""
	shifted   = points - center
	spherical = shifted / np.linalg.norm(shifted, axis=1, keepdims=True)

	cloud = pv.PolyData(spherical)
	cloud.point_data["orig_idx"] = np.arange(len(points))

	mesh = cloud.delaunay_3d().extract_surface()
	idx  = mesh.point_data["orig_idx"]
	mesh.points = points[idx]
	return mesh, idx


def integrate_force(mesh, field):
	"""Integrate a vector cell field over the mesh surface."""
	return mesh.integrate_data().cell_data[field][0]

# ============================================================
# Build point cloud
# ============================================================

rng    = np.random.default_rng(SEED)
points = sample_cylinder_surface(R, H, N_POINTS, top_cap=True, bottom_cap=True, rng=rng)
cloud  = pv.PolyData(points)

cloud["velocities"] = np.tile(VELOCITY, (cloud.n_points, 1))
cloud["pressures"]  = np.full(cloud.n_points, PRESSURE)

print(f"Input points : {cloud.n_points}")

# ============================================================
# Surface reconstruction
# ============================================================

center     = np.array([0.0, 0.0, H / 2.0])
mesh, idx  = reconstruct_surface(points, center)

mesh["velocities"] = cloud["velocities"][idx]
mesh["pressures"]  = cloud["pressures"][idx]

print(f"Vertices     : {mesh.n_points}")
print(f"Triangles    : {mesh.n_cells}")
print(f"Open edges   : {mesh.n_open_edges}")

# ============================================================
# Prepare cell data and normals
# ============================================================

mesh = mesh.ptc().compute_normals(
	cell_normals=True,
	point_normals=False,
	consistent_normals=True,
	auto_orient_normals=True,
)

# ============================================================
# Geometry diagnostics
# ============================================================

expected_volume = np.pi * R**2 * H
volume_error    = (mesh.volume - expected_volume) / expected_volume * 100

areas      = mesh.compute_cell_sizes()["Area"]
net_normal = (mesh.cell_data["Normals"] * areas[:, None]).sum(axis=0)

print(f"\nExpected volume : {expected_volume:.6f}")
print(f"Mesh volume     : {mesh.volume:.6f}  ({volume_error:+.3f}%)")
print(f"Net normal      : {net_normal}")

# ============================================================
# Force integration
# ============================================================

v = mesh.cell_data["velocities"]
p = mesh.cell_data["pressures"]
n = mesh.cell_data["Normals"]

mesh.cell_data["pressure_force"] = p[:, None] * n
mesh.cell_data["momentum_force"] = RHO * v * np.einsum("ij,ij->i", v, n)[:, None] #some fucked up dot product for the whole thing v dot n but for many rows
mesh.cell_data["total_force"]    = mesh.cell_data["pressure_force"] + mesh.cell_data["momentum_force"]

Fp = integrate_force(mesh, "pressure_force")
Fm = integrate_force(mesh, "momentum_force")
F  = integrate_force(mesh, "total_force")

print(f"\nPressure force : {Fp}")
print(f"Momentum force : {Fm}")
print(f"Total force    : {F}  (|F| = {np.linalg.norm(F):.6e})")

# ============================================================
# Visualisation
# ============================================================

def visualize(mesh, normal_scale=0.2):
	pl = pv.Plotter()
	pl.add_mesh(mesh, color="lightgray", show_edges=True, edge_color="black")
	pl.add_arrows(mesh.cell_centers().points, mesh.cell_data["Normals"], mag=normal_scale, color="red")
	pl.add_axes()
	pl.show()

visualize(mesh)
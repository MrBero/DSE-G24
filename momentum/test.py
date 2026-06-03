import numpy as np
import open3d as o3d


def generate_lateral_surface_points(R, H, num_points):
	theta = np.random.uniform(0, 2 * np.pi, num_points)
	z = np.random.uniform(0, H, num_points)
	x = R * np.cos(theta)
	y = R * np.sin(theta)
	return np.column_stack((x, y, z))


def reconstruct_ball_pivot(points, center, ball_radii=None):
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)

	pts = np.asarray(pcd.points)
	normals = pts - np.asarray(center)
	normals[:, 2] = 0.0  # zero out Z component, normals are purely radial
	normals /= np.linalg.norm(normals, axis=1, keepdims=True)
	pcd.normals = o3d.utility.Vector3dVector(normals)

	if ball_radii is None:
		avg = np.mean(pcd.compute_nearest_neighbor_distance())
		ball_radii = [avg * 16]

	mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
		pcd, o3d.utility.DoubleVector(ball_radii)
	)
	mesh.compute_triangle_normals()

	print(f"Vertices : {len(mesh.vertices)}")
	print(f"Triangles: {len(mesh.triangles)}")
	print(f"Watertight: {mesh.is_watertight()}")
	return mesh

def reconstruct_alpha_shape(points, alpha=None):
    """
    Reconstructs a mesh using the Alpha Shapes method.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if alpha is None:
        # Default heuristic for alpha parameter
        alpha = np.mean(pcd.compute_nearest_neighbor_distance()) * 2.0

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_triangle_normals()

    print("--- Alpha Shape ---")
    print(f"Vertices : {len(mesh.vertices)}")
    print(f"Triangles: {len(mesh.triangles)}")
    print(f"Watertight: {mesh.is_watertight()}")
    return mesh


# def reconstruct_delaunay_cylinder(points, center):
#     """
#     Reconstructs a mesh by unrolling the point cloud onto a 2D cylinder (theta, z),
#     performing 2D Delaunay triangulation, and mapping back to 3D.
#     """
#     pts = np.asarray(points) - np.asarray(center)
    
#     theta = np.arctan2(pts[:, 1], pts[:, 0])
#     z = pts[:, 2]

#     # Project to 2D
#     points_2d = np.column_stack((theta, z))
#     tri = Delaunay(points_2d)

#     # Filter triangles crossing the seam
#     triangles = []
#     for simplex in tri.simplices:
#         theta_vals = theta[simplex]
#         if np.max(theta_vals) - np.min(theta_vals) < np.pi:
#             triangles.append(simplex)

#     mesh = o3d.geometry.TriangleMesh()
#     mesh.vertices = o3d.utility.Vector3dVector(points)
#     mesh.triangles = o3d.utility.Vector3iVector(triangles)
#     mesh.compute_triangle_normals()

#     print("--- Delaunay Cylinder ---")
#     print(f"Vertices : {len(mesh.vertices)}")
#     print(f"Triangles: {len(mesh.triangles)}")
#     print(f"Watertight: {mesh.is_watertight()}")
#     return mesh
# def reconstruct_delaunay_sphere(points, center):
#     """
#     Reconstructs a mesh by unrolling the point cloud onto a 2D sphere (theta, phi),
#     performing 2D Delaunay triangulation, and mapping back to 3D.
#     """
#     pts = np.asarray(points) - np.asarray(center)
    
#     r = np.linalg.norm(pts, axis=1)
#     r[r == 0] = 1e-8  # Prevent division by zero
    
#     theta = np.arctan2(pts[:, 1], pts[:, 0])
#     phi = np.arccos(pts[:, 2] / r)

#     # Project to 2D
#     points_2d = np.column_stack((theta, phi))
#     tri = Delaunay(points_2d)

#     # Filter triangles crossing the seam
#     triangles = []
#     for simplex in tri.simplices:
#         theta_vals = theta[simplex]
#         if np.max(theta_vals) - np.min(theta_vals) < np.pi:
#             triangles.append(simplex)

#     mesh = o3d.geometry.TriangleMesh()
#     mesh.vertices = o3d.utility.Vector3dVector(points)
#     mesh.triangles = o3d.utility.Vector3iVector(triangles)
#     mesh.compute_triangle_normals()

#     print("--- Delaunay Sphere ---")
#     print(f"Vertices : {len(mesh.vertices)}")
#     print(f"Triangles: {len(mesh.triangles)}")
#     print(f"Watertight: {mesh.is_watertight()}")
#     return mesh


def visualize(points, mesh, show_points=True, show_mesh=True,
              show_face_normals=True, show_wireframe=False):
	geometries = []

	if show_points:
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(points)
		z_vals = points[:, 2]
		z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-10)
		pcd.colors = o3d.utility.Vector3dVector(np.column_stack([
			z_norm,
			1.0 - z_norm,
			np.ones(len(z_norm)) * 0.8,
		]))
		geometries.append(pcd)

	if show_mesh:
		render_mesh = o3d.geometry.TriangleMesh(mesh)
		render_mesh.compute_vertex_normals()
		render_mesh.paint_uniform_color([0.72, 0.72, 0.85])
		geometries.append(render_mesh)

	if show_face_normals:
		verts     = np.asarray(mesh.vertices)
		tris      = np.asarray(mesh.triangles)
		tri_norms = np.asarray(mesh.triangle_normals)

		centroids = verts[tris].mean(axis=1)
		scale     = np.mean(np.linalg.norm(verts[tris[:, 1]] - verts[tris[:, 0]], axis=1)) * 1.5

		step = max(1, len(centroids) // 1000)
		c = centroids[::step]
		n = tri_norms[::step]

		line_pts, line_idx = [], []
		for i, (p, nv) in enumerate(zip(c, n)):
			line_pts.extend([p, p + nv * scale])
			line_idx.append([2 * i, 2 * i + 1])

		ls = o3d.geometry.LineSet()
		ls.points = o3d.utility.Vector3dVector(line_pts)
		ls.lines  = o3d.utility.Vector2iVector(line_idx)
		ls.paint_uniform_color([1.0, 0.4, 0.1])
		geometries.append(ls)

	if show_wireframe:
		wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
		wire.paint_uniform_color([0.2, 0.2, 0.2])
		geometries.append(wire)

	geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5))

	o3d.visualization.draw_geometries(
		geometries,
		window_name="Cylinder Reconstruction",
		width=1024,
		height=768,
	)


# --- Usage ---
R, H = 5.0, 10.0

points = generate_lateral_surface_points(R=R, H=H, num_points=80)
#mesh   = reconstruct_ball_pivot(points, center = [0,0,H/2], ball_radii = [2*R])
mesh = reconstruct_alpha_shape(points, alpha=2*R)

#o3d.io.write_triangle_mesh("cylinder.ply", mesh)
visualize(points, mesh, show_points=True, show_mesh=True, show_face_normals=True)
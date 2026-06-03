import pyvista as pv
import open3d as o3d
import numpy as np

# --- Parameters ---
R, H = 3, 10
RHO = 1.225
N_POINTS = 80

# --- 1. Generate Points on Lateral Cylinder Surface ---
import numpy as np

def cylinder_surface_points(R, H, n, top_cap=False, bottom_cap=False):
    np.random.seed(42)
    
    # 1. Calculate surface areas
    area_lat = 2 * np.pi * R * H
    area_top = np.pi * R**2 if top_cap else 0.0
    area_bot = np.pi * R**2 if bottom_cap else 0.0
    total_area = area_lat + area_top + area_bot
    
    # 2. Distribute point counts proportionally
    n_lat = int(n * (area_lat / total_area))
    n_top = int(n * (area_top / total_area)) if top_cap else 0
    n_bot = int(n * (area_bot / total_area)) if bottom_cap else 0
    
    # Assign any rounding remainder to the lateral surface
    n_lat += n - (n_lat + n_top + n_bot)
    
    points = []
    
    # 3. Generate Lateral Points
    if n_lat > 0:
        theta_lat = np.random.uniform(0, 2 * np.pi, n_lat)
        z_lat     = np.random.uniform(0, H, n_lat)
        points.append(np.column_stack([R * np.cos(theta_lat), R * np.sin(theta_lat), z_lat]))
        
    # 4. Generate Top Cap Points (z = H)
    if top_cap and n_top > 0:
        r_top     = R * np.sqrt(np.random.uniform(0, 1, n_top))
        theta_top = np.random.uniform(0, 2 * np.pi, n_top)
        points.append(np.column_stack([r_top * np.cos(theta_top), r_top * np.sin(theta_top), np.full(n_top, H)]))
        
    # 5. Generate Bottom Cap Points (z = 0)
    if bottom_cap and n_bot > 0:
        r_bot     = R * np.sqrt(np.random.uniform(0, 1, n_bot))
        theta_bot = np.random.uniform(0, 2 * np.pi, n_bot)
        points.append(np.column_stack([r_bot * np.cos(theta_bot), r_bot * np.sin(theta_bot), np.zeros(n_bot)]))
        
    return np.vstack(points)


points = cylinder_surface_points(R, H, N_POINTS, top_cap=False, bottom_cap=False)

# --- 2. Attach Uniform Flow Properties to Original Cloud ---
cloud = pv.PolyData(points)
cloud.point_data["velocities"] = np.tile([0, 0.0, 0.0], (len(points), 1))
cloud.point_data["pressures"]  = np.full(len(points), 20.0)

# --- 3. Open3D BPA Reconstruction to PyVista ---
def reconstruct_ball_pivot_to_pyvista(points, center, ball_radii=None):
    # Execute Open3D BPA
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pts = np.asarray(pcd.points)
    normals = pts - np.asarray(center)
    normals[:, 2] = 0.0  # Zero out Z component for radial normals
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    if ball_radii is None:
        avg = np.mean(pcd.compute_nearest_neighbor_distance())
        ball_radii = [avg * 1.5, avg * 3, avg * 6]

    o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(ball_radii)
    )
    
    print("\n--- Mesh Generation Statistics ---")
    print(f"Open3D Vertices : {len(o3d_mesh.vertices)}")
    print(f"Open3D Triangles: {len(o3d_mesh.triangles)}")
    print(f"Watertight      : {o3d_mesh.is_watertight()}")

    # PyVista Conversion
    vertices = np.asarray(o3d_mesh.vertices)
    triangles = np.asarray(o3d_mesh.triangles)
    faces = np.column_stack((np.full(len(triangles), 3), triangles)).flatten()
    
    return pv.PolyData(vertices, faces)

print("Reconstructing surface using Open3D BPA...")
mesh = reconstruct_ball_pivot_to_pyvista(points, center=[0.0, 0.0, H/2], ball_radii=[R])

# --- 4. Transfer Data & Compute Normals ---
# Sample point data from the original cloud to the new BPA mesh vertices
mesh = mesh.sample(cloud)

# Convert point data to cell data
mesh = mesh.ptc()                                                           
mesh = mesh.compute_normals(cell_normals=True, point_normals=False,
                            consistent_normals=True)

v = mesh.cell_data["velocities"]   # (N, 3)
p = mesh.cell_data["pressures"]    # (N,)
n = mesh.cell_data["Normals"]      # (N, 3)

# --- 5. Compute Aerodynamic Integrand per Cell ---
v_dot_n = np.einsum("ij,ij->i", v, n)[:, np.newaxis]   # (N, 1)
mesh.cell_data["integrand"] = RHO * v * v_dot_n + p[:, np.newaxis] * n

# --- 6. Integrate Over Surface (area-weighted) ---
F = mesh.integrate_data().cell_data["integrand"][0]

print("\n--- Integration Results ---")
for label, val in zip("xyz", F):
    print(f"  F{label}: {val:.4f} N")

# --- 7. Visualise ---
def visualize(mesh, normal_scale=0.1):
    pl = pv.Plotter()
    
    # Render base mesh
    pl.add_mesh(mesh, color="lightgray", show_edges=True,
                edge_color="black", lighting=True)
    
    # Compute surface (cell) normals
    mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    
    # Extract cell centers for vector origin points
    centers = mesh.cell_centers()
    
    # Render normal vectors
    pl.add_arrows(centers.points, mesh.cell_data["Normals"], 
                  color="red", mag=normal_scale)
    
    pl.add_axes()
    pl.show()
visualize(mesh)
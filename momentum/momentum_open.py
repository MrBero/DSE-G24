import numpy as np
import open3d as o3d
import pyvista as pv

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_cylinder_uniform(R: float, H: float, n: int) -> np.ndarray:
    """Sample staggered uniform points on the lateral surface of a cylinder."""
    n_cols = max(1, round(np.sqrt(n * 2 * np.pi * R / H)))
    n_rows = max(1, round(n / n_cols))
    
    idx = np.arange(n_rows * n_cols)
    row_idx = idx // n_cols
    col_idx = idx % n_cols
    
    # Apply a half-step phase shift to alternating rows
    theta_shift = (row_idx % 2) * (np.pi / n_cols)
    theta = (2 * np.pi * col_idx / n_cols) + theta_shift
    
    z = H * row_idx / max(n_rows - 1, 1)
    
    return np.c_[R * np.cos(theta), R * np.sin(theta), z]

# ---------------------------------------------------------------------------
# BPA Reconstruction
# ---------------------------------------------------------------------------
def reconstruct_surface_bpa(
    points: np.ndarray,
    L: float,
    center: np.ndarray,
    radii_factors: list[float] = [1.0]
) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Compute normals directly outward from the provided midpoint
    normals = points - center
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    radii = o3d.utility.DoubleVector([L * f for f in radii_factors])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    mesh.compute_vertex_normals()
    return mesh, pcd

def o3d_to_pyvista(mesh: o3d.geometry.TriangleMesh) -> pv.PolyData:
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    faces_pv = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces])
    return pv.PolyData(verts, faces_pv.ravel())

# ---------------------------------------------------------------------------
# Force Integration
# ---------------------------------------------------------------------------
def surface_force_bpa(
    points: np.ndarray,
    center: np.ndarray,
    velocity: np.ndarray, # Shape (N, 3)
    pressure: np.ndarray, # Shape (N,)
    L: float,
    rho: float = 1.225,
    radii_factors: list[float] = [1.0]
) -> tuple[np.ndarray, pv.PolyData, o3d.geometry.PointCloud]:
    
    mesh_o3d, pcd = reconstruct_surface_bpa(points, L, center, radii_factors)
    mesh_pv = o3d_to_pyvista(mesh_o3d).compute_normals(
        cell_normals=True, point_normals=False, auto_orient_normals=True
    )

    # 1. Assign local values to point data
    mesh_pv.point_data["velocity"] = velocity
    mesh_pv.point_data["pressure"] = pressure

    # 2. Interpolate point data to cell data for alignment with cell normals
    mesh_pv = mesh_pv.point_data_to_cell_data()

    n = mesh_pv.cell_data["Normals"]
    v = mesh_pv.cell_data["velocity"]
    p = mesh_pv.cell_data["pressure"]

    # 3. Compute dot product and force per cell area
    v_dot_n = np.sum(v * n, axis=1, keepdims=True)
    mesh_pv.cell_data["total_force"] = (p[:, np.newaxis] * n) + (rho * v * v_dot_n)
    
    # 4. Integrate total force over surface area
    F = mesh_pv.integrate_data().cell_data["total_force"][0]
    
    return F, mesh_pv, pcd

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    R, H = 3.0, 10.0
    center = np.array([0.0, 0.0, H / 2.0])
    
    points = sample_cylinder_uniform(R, H, n=3000)
    num_points = points.shape[0]

    # Generate pointwise arrays (using the original global values as an example)
    velocities = np.zeros((num_points, 3))
    velocities[:, 0] = 10.0 
    
    pressures = np.full(num_points, 20.0)

    F, mesh, pcd = surface_force_bpa(
        points=points,
        center=center,
        velocity=velocities,
        pressure=pressures,
        L=0.5, 
        radii_factors=[0.5, 1.0, 2.0]
    )

    print(f"Force vector : {F}")
    print(f"|F|          : {np.linalg.norm(F):.6e}")

    # Minimal visualization
    pl = pv.Plotter()
    pl.add_mesh(mesh, scalars="pressure", cmap="viridis", show_edges=True)
    pl.add_points(np.asarray(pcd.points), color="navy", point_size=5)
    pl.show()
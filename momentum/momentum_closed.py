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

    mesh     = cloud.delaunay_3d().extract_surface(algorithm='dataset_surface')
    idx      = mesh.point_data["orig_idx"]
    mesh.points = points[idx]

    return mesh, idx


def compute_forces(
    mesh: pv.PolyData,
    velocity: np.ndarray,
    pressure: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, pv.PolyData]:
    """
    Integrate pressure and momentum forces over a triangulated surface.

    Requires velocity and pressure to be passed as point data arrays.
    Returns total force as a (3,) array and the computed mesh.
    """
    mesh = mesh.compute_normals(
        cell_normals=True,
        point_normals=False,
        consistent_normals=True,
        auto_orient_normals=True,
    )

    mesh.point_data["velocity"] = velocity
    mesh.point_data["pressure"] = pressure

    mesh = mesh.point_data_to_cell_data()

    n = mesh.cell_data["Normals"]
    v = mesh.cell_data["velocity"]
    p = mesh.cell_data["pressure"]

    v_dot_n = (v * n).sum(axis=1, keepdims=True)

    mesh.cell_data["pressure_force"] = p[:, np.newaxis] * n
    mesh.cell_data["momentum_force"] = rho * v * v_dot_n
    mesh.cell_data["total_force"]    = mesh.cell_data["pressure_force"] + mesh.cell_data["momentum_force"]

    F = mesh.integrate_data().cell_data["total_force"][0]
    
    return F, mesh


def surface_force(
    points: np.ndarray,
    center: np.ndarray,
    velocity: np.ndarray,
    pressure: np.ndarray,
    rho: float = 1.225,
) -> tuple[np.ndarray, pv.PolyData]:
    """
    Compute the net aerodynamic force on a surface defined by a point cloud.

    Parameters
    ----------
    points   : (N, 3) surface point cloud
    center   : (3,)  interior point used for surface reconstruction
    velocity : (N, 3) point-wise velocity vectors
    pressure : (N,) point-wise static pressure
    rho      : float fluid density (default 1.225 kg/m^3)

    Returns
    -------
    F    : (3,) net force vector [Fx, Fy, Fz]
    mesh : pv.PolyData computed mesh with assigned force arrays
    """
    points   = np.asarray(points,   dtype=float)
    center   = np.asarray(center,   dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    pressure = np.asarray(pressure, dtype=float)

    mesh, idx = reconstruct_surface(points, center)
    
    mapped_velocity = velocity[idx]
    mapped_pressure = pressure[idx]

    return compute_forces(mesh, mapped_velocity, mapped_pressure, rho)


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

    points = sample_cylinder(R, H, n=800, rng=np.random.default_rng(67))
    center = np.array([0.0, 0.0, H / 2.0])
    
    num_points = points.shape[0]

    velocities = np.zeros((num_points, 3))
    velocities[:, 0] = 10.0
    
    pressures = np.full(num_points, 20.0)

    F, mesh = surface_force(
        points=points,
        center=center,
        velocity=velocities,
        pressure=pressures,
        rho=1.225,
    )

    print(f"Force vector : {F}")
    print(f"|F|          : {np.linalg.norm(F):.6e}")
    print(f"Expected vol : {np.pi * R**2 * H:.4f}")

    pl = pv.Plotter()
    
    # Render mesh colored by interpolated pressure
    pl.add_mesh(mesh, scalars="pressure", cmap="viridis", show_edges=True, opacity=0.8)
    
    # Generate and render velocity vector glyphs
    mesh.set_active_vectors("velocity")
    #arrows = mesh.glyph(orient="velocity", scale="velocity", factor=0.1)
    #pl.add_mesh(arrows, color="white")
    
    # Plot settings
    pl.add_axes()
    pl.show()
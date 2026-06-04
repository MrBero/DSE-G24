import pyvista as pv
import numpy as np

# ============================================================
# Parameters
# ============================================================

R = 3.0
H = 10.0
RHO = 1.225
N_POINTS = 800
ALPHA = 1

# ============================================================
# Generate random points on cylinder surface
# ============================================================

np.random.seed(67)

def cylinder_surface_points(R, H, n, top_cap=False, bottom_cap=False):
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

    # 3. Generate lateral surface points
    if n_lat > 0:
        theta_lat = np.random.uniform(0, 2 * np.pi, n_lat)
        z_lat = np.random.uniform(0, H, n_lat)

        points.append(
            np.column_stack(
                [
                    R * np.cos(theta_lat),
                    R * np.sin(theta_lat),
                    z_lat,
                ]
            )
        )

    # 4. Generate top cap points
    if top_cap and n_top > 0:
        r_top = R * np.sqrt(np.random.uniform(0, 1, n_top))
        theta_top = np.random.uniform(0, 2 * np.pi, n_top)

        points.append(
            np.column_stack(
                [
                    r_top * np.cos(theta_top),
                    r_top * np.sin(theta_top),
                    np.full(n_top, H),
                ]
            )
        )

    # 5. Generate bottom cap points
    if bottom_cap and n_bot > 0:
        r_bot = R * np.sqrt(np.random.uniform(0, 1, n_bot))
        theta_bot = np.random.uniform(0, 2 * np.pi, n_bot)

        points.append(
            np.column_stack(
                [
                    r_bot * np.cos(theta_bot),
                    r_bot * np.sin(theta_bot),
                    np.zeros(n_bot),
                ]
            )
        )

    return np.vstack(points)

# ============================================================
# Generate point cloud
# ============================================================

points = cylinder_surface_points(
    R,
    H,
    N_POINTS, top_cap=True, bottom_cap=True,
)

# ============================================================
# Create point cloud and attach fields
# ============================================================

cloud = pv.PolyData(points)

print(f"Input points : {cloud.n_points}")

cloud.plot(
    render_points_as_spheres=True,
    point_size=10,
    show_axes=True,
)

cloud["velocities"] = np.tile(
    [10.0, 0.0, 0.0],
    (cloud.n_points, 1),
)

cloud["pressures"] = np.full(
    cloud.n_points,
    20.0,
)

# ============================================================
# Surface reconstruction using alpha shapes
# ============================================================

print("Reconstructing surface...")

mesh = (
    cloud
    .delaunay_3d(alpha=ALPHA)
    .extract_surface()
    .triangulate()
    .connectivity("largest")
    .clean()
)

# Transfer point data onto reconstructed mesh
mesh = mesh.interpolate(cloud)

print(f"Vertices     : {mesh.n_points}")
print(f"Triangles    : {mesh.n_cells}")
print(f"Open edges   : {mesh.n_open_edges}")

# ============================================================
# Convert point data -> cell data
# ============================================================

mesh = mesh.ptc()

# ============================================================
# Compute normals
# ============================================================

mesh = mesh.compute_normals(
    cell_normals=True,
    point_normals=False,
    consistent_normals=True,
    auto_orient_normals=True,
)

# ============================================================
# Diagnostics
# ============================================================

expected_volume = np.pi * R**2 * H

print("\nGeometry diagnostics")
print("--------------------")
print(f"Expected volume : {expected_volume:.6f}")
print(f"Mesh volume     : {mesh.volume:.6f}")
print(
    f"Volume error    : "
    f"{(mesh.volume - expected_volume) / expected_volume * 100:.3f}%"
)

areas = mesh.compute_cell_sizes()["Area"]
normals = mesh.cell_data["Normals"]

net_normal = np.sum(normals * areas[:, None], axis=0)

print("\nIntegral of n dA")
print(net_normal)

# ============================================================
# Force integration
# ============================================================

v = mesh.cell_data["velocities"]
p = mesh.cell_data["pressures"]
n = mesh.cell_data["Normals"]

v_dot_n = np.einsum("ij,ij->i", v, n)[:, None]

# Pressure contribution
mesh.cell_data["pressure_force"] = p[:, None] * n
Fp = mesh.integrate_data().cell_data["pressure_force"][0]

# Momentum contribution
mesh.cell_data["momentum_force"] = RHO * v * v_dot_n
Fm = mesh.integrate_data().cell_data["momentum_force"][0]

# Total force
mesh.cell_data["total_force"] = (
    mesh.cell_data["pressure_force"]
    + mesh.cell_data["momentum_force"]
)

F = mesh.integrate_data().cell_data["total_force"][0]

# ============================================================
# Results
# ============================================================

print("\nPressure force")
print("--------------")
print(Fp)

print("\nMomentum force")
print("--------------")
print(Fm)

print("\nTotal force")
print("-----------")
print(F)

print(f"\n|F| = {np.linalg.norm(F):.6e}")

# ============================================================
# Visualisation
# ============================================================

def visualize(mesh, normal_scale=0.2):
    pl = pv.Plotter()

    pl.add_mesh(
        mesh,
        color="lightgray",
        show_edges=True,
        edge_color="black",
    )

    centers = mesh.cell_centers()

    pl.add_arrows(
        centers.points,
        mesh.cell_data["Normals"],
        mag=normal_scale,
        color="red",
    )

    pl.add_axes()
    pl.show()

visualize(mesh)
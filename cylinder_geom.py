"""
cylinder_geom.py
----------------
Cylinder geometry for the wind-force pipeline:

    calculate_wake_cylinder_parameters   -- fit a tilted cylinder around a building
    generate_cylindrical_sampling_coordinates  -- drone / GPR training points on the surface
    generate_momentum_integration_mesh   -- fine triangulated surface for force integration
    visualize_points                     -- quick point-cloud preview
"""

from __future__ import annotations
import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist

try:
    import triangle as tr
except ImportError:
    tr = None

GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))   # ~2.3999 rad  (Vogel spiral)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_wake_cylinder_parameters(stl_mesh, r_factor, h_factor, v_inf,
                                       tilt_deg=0.0):
    """
    Fit a tilted cylinder that fully encloses a building and extends into its wake.

    The cylinder axis leans downstream (in the v_inf direction) with height,
    controlled by tilt_deg.  Raises ValueError if the building sticks out.

    Parameters
    ----------
    stl_mesh  : trimesh.Trimesh  (already scaled to metres)
    r_factor  : float  radius = r_factor × footprint circumradius
    h_factor  : float  height = h_factor × building height
    v_inf     : (3,)   free-stream velocity vector
    tilt_deg  : float  downstream wake tilt in degrees

    Returns
    -------
    center_bot, center_top : (3,) arrays
    radius                 : float
    """
    verts = np.asarray(stl_mesh.vertices)

    # Horizontal wind direction (for tilt axis)
    v_xy = np.array([v_inf[0], v_inf[1], 0.0], dtype=float)
    norm = np.linalg.norm(v_xy)
    if norm < 1e-12:
        raise ValueError("v_inf has no horizontal component.")
    v_xy /= norm

    # Smallest enclosing circle of the building footprint
    xy = np.unique(verts[:, :2], axis=0)
    hull_xy = xy[ConvexHull(xy).vertices]
    footprint_r = 0.5 * cdist(hull_xy, hull_xy).max()
    radius = r_factor * footprint_r

    # Vertical extents and cylinder dimensions
    z_bot = verts[:, 2].min()
    z_top = verts[:, 2].max()
    z_mid = 0.5 * (z_bot + z_top)
    height = h_factor * (z_top - z_bot)

    # XY centre of bounding box
    xy_ctr = np.array([
        0.5 * (verts[:, 0].min() + verts[:, 0].max()),
        0.5 * (verts[:, 1].min() + verts[:, 1].max()),
    ])

    tilt = np.tan(np.radians(tilt_deg))

    def ring_center(z):
        offset = xy_ctr + tilt * (z - z_mid) * v_xy[:2]
        return np.append(offset, z)

    c_bot = ring_center(z_bot)
    c_top = ring_center(z_bot + height)

    # Validate: every building vertex must be inside the cylinder
    axis  = c_top - c_bot
    axis2 = np.dot(axis, axis)
    diff  = verts - c_bot
    t     = np.clip((diff @ axis) / axis2, 0.0, 1.0)
    closest = c_bot + t[:, None] * axis
    r_dist  = np.linalg.norm(verts - closest, axis=1)
    t_full  = (diff @ axis) / axis2           # unclipped, for range check
    outside = (r_dist > radius) & (t_full >= 0.0) & (t_full <= 1.0)
    if outside.any():
        raise ValueError(
            f"{outside.sum()} building vertices lie outside the cylinder. "
            f"Increase r_factor ({r_factor}) or h_factor ({h_factor})."
        )

    return c_bot, c_top, radius


def generate_cylindrical_sampling_coordinates(
    center_bot, center_top, radius,
    z_clearance=0.1,
    side_points=0,
    top_points=0,
    bot_points=0,
) -> np.ndarray:
    """
    Sampling points (drone locations) on / near the cylinder surface.

    Side points use a golden-angle spiral along the axis so they are
    quasi-uniformly distributed.  Cap points use a Vogel disk.

    Parameters
    ----------
    center_bot, center_top : (3,) cylinder axis endpoints
    radius      : float
    z_clearance : vertical gap between the bottom cap and the first side point
    side_points : number of points on the cylindrical wall
    top_points  : number of points on the top cap
    bot_points  : number of points on the bottom cap

    Returns
    -------
    (N, 3) array  where N = side_points + top_points + bot_points
    """
    c_bot = np.asarray(center_bot, dtype=float)
    c_top = np.asarray(center_top, dtype=float)
    axis  = c_top - c_bot
    axis_unit = axis / np.linalg.norm(axis)

    # Shift bottom up by z_clearance along the (oblique) axis
    dz_frac = z_clearance / axis_unit[2]
    c_bot_shifted = c_bot + axis_unit * dz_frac

    parts = []

    if side_points > 0:
        i      = np.arange(side_points)
        alpha  = i / (side_points - 1) if side_points > 1 else np.array([0.5])
        theta  = i * GOLDEN_ANGLE
        cx = c_bot_shifted[0] + alpha * (c_top[0] - c_bot_shifted[0])
        cy = c_bot_shifted[1] + alpha * (c_top[1] - c_bot_shifted[1])
        cz = c_bot_shifted[2] + alpha * (c_top[2] - c_bot_shifted[2])
        parts.append(np.stack([cx + radius*np.cos(theta),
                                cy + radius*np.sin(theta), cz], axis=1))

    def vogel_disk(cx, cy, z, n):
        i = np.arange(n)
        r = radius * np.sqrt((i + 0.5) / n)
        t = i * GOLDEN_ANGLE
        return np.stack([cx + r*np.cos(t), cy + r*np.sin(t), np.full(n, z)], axis=1)

    if bot_points > 0:
        parts.append(vogel_disk(*c_bot_shifted[:2], c_bot_shifted[2], bot_points))
    if top_points > 0:
        parts.append(vogel_disk(*c_top[:2], c_top[2], top_points))

    return np.vstack(parts)


def generate_momentum_integration_mesh(
    center_bot, center_top, radius,
    total_points=100_000,
    cap_bottom=False,
    cap_top=False,
) -> pv.PolyData:
    """
    Triangulated surface mesh of an oblique cylinder for momentum integration.

    Points are distributed proportionally to surface area across wall and caps.
    Lateral rings are staggered by half a step to improve triangle quality.
    Cap interiors use a Vogel spiral with constrained Delaunay triangulation
    (falls back to plain Delaunay if the `triangle` package is not installed).

    Parameters
    ----------
    center_bot, center_top : (3,) cylinder axis endpoints
    radius       : float
    total_points : target total vertex count
    cap_bottom   : include bottom cap
    cap_top      : include top cap

    Returns
    -------
    pv.PolyData  (triangles only)
    """
    c_bot = np.asarray(center_bot, dtype=float)
    c_top = np.asarray(center_top, dtype=float)
    cx_b, cy_b, z_bot = c_bot
    cx_t, cy_t, z_top = c_top
    L = float(np.linalg.norm(c_top - c_bot))

    n_caps = int(cap_bottom) + int(cap_top)
    lat_area = 2 * np.pi * radius * L
    cap_area = np.pi * radius**2
    total_area = lat_area + n_caps * cap_area

    s = np.sqrt(total_area / total_points)        # target spacing
    T = max(4, round(2 * np.pi * radius / s))     # points per ring
    Z = max(1, round(L / s))                      # rings along axis

    dtheta = 2 * np.pi / T
    # Stagger odd rings by half a step
    base_theta = np.linspace(0, 2*np.pi, T, endpoint=False)
    shifts = (np.arange(Z + 1) % 2) * (dtheta / 2)
    thetas_2d = base_theta[None, :] + shifts[:, None]   # (Z+1, T)

    alpha  = np.linspace(0, 1, Z + 1)
    cx_r = cx_b + alpha * (cx_t - cx_b)
    cy_r = cy_b + alpha * (cy_t - cy_b)
    z_r  = z_bot + alpha * (z_top - z_bot)

    # --- Lateral vertices (Z+1 rings × T points) ---
    lat_x = cx_r[:, None] + radius * np.cos(thetas_2d)
    lat_y = cy_r[:, None] + radius * np.sin(thetas_2d)
    lat_z = np.tile(z_r[:, None], (1, T))
    lat_pts = np.stack([lat_x.ravel(), lat_y.ravel(), lat_z.ravel()], axis=1)

    def lat_idx(ring, j):
        return ring * T + (j % T)

    # --- Lateral faces (staggered triangles) ---
    faces = []
    for i in range(Z):
        for j in range(T):
            jn = (j + 1) % T
            a, b = lat_idx(i, j), lat_idx(i, jn)
            c, d = lat_idx(i+1, j), lat_idx(i+1, jn)
            if i % 2 == 0:
                faces += [3, a, b, c,  3, c, b, d]
            else:
                faces += [3, a, b, d,  3, c, a, d]

    all_pts = [lat_pts]

    # --- Cap builder ---
    def add_cap(ring_i, cx, cy, z, inward):
        # Vogel interior points (slightly shrunk to stay inside the rim)
        n_int = max(0, round(cap_area / s**2) - T)
        if n_int > 0:
            k = np.arange(n_int)
            r = 0.97 * radius * np.sqrt((k + 0.5) / n_int)
            t = k * GOLDEN_ANGLE
            int_pts = np.stack([cx + r*np.cos(t), cy + r*np.sin(t),
                                np.full(n_int, z)], axis=1)
        else:
            int_pts = np.empty((0, 3))

        # 2-D coords for triangulation: rim first, then interior
        rim_2d = np.stack([
            cx + radius * np.cos(base_theta + shifts[ring_i]),
            cy + radius * np.sin(base_theta + shifts[ring_i]),
        ], axis=1)
        pts_2d = np.vstack([rim_2d, int_pts[:, :2]]) if n_int > 0 else rim_2d

        if tr is not None:
            segs = np.array([[i, (i+1) % T] for i in range(T)], dtype=np.int_)
            result = tr.triangulate({"vertices": pts_2d, "segments": segs}, "p")
            simplices = result["triangles"]
        else:
            simplices = Delaunay(pts_2d).simplices

        int_offset = sum(p.shape[0] for p in all_pts)
        if n_int > 0:
            all_pts.append(int_pts)

        def g(local):   # local index → global index
            return lat_idx(ring_i, local) if local < T else int_offset + (local - T)

        for s_ in simplices:
            a, b, c = g(s_[0]), g(s_[1]), g(s_[2])
            faces.extend([3, a, c, b] if inward else [3, a, b, c])

    if cap_bottom:
        add_cap(0, cx_b, cy_b, z_bot, inward=True)
    if cap_top:
        add_cap(Z, cx_t, cy_t, z_top, inward=False)

    return pv.PolyData(np.vstack(all_pts), np.array(faces, dtype=np.int_))


def visualize_points(pts: np.ndarray, point_size=5, color="cyan") -> None:
    """Quick point-cloud preview with a ground plane at z=0."""
    cloud = pv.PolyData(pts)
    pl = pv.Plotter(window_size=[900, 700])
    pl.set_background("black")
    pl.add_points(cloud, color=color, point_size=point_size,
                  render_points_as_spheres=True)
    b = cloud.bounds
    cx, cy = (b[0]+b[1])/2, (b[2]+b[3])/2
    size = max(b[1]-b[0], b[3]-b[2]) * 3
    pl.add_mesh(pv.Plane(center=(cx, cy, 0), direction=(0,0,1),
                         i_size=size, j_size=size), color="gray", opacity=0.2)
    pl.add_axes()
    pl.show()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import trimesh

    STL_PATH = r"input_stls/Aerospecial_building4.stl"
    V_INF    = np.array([0.0, 13.6, 0.0])

    stl = trimesh.load_mesh(STL_PATH)
    stl.apply_scale(1e-3)

    bot, top, radius = calculate_wake_cylinder_parameters(
        stl, r_factor=3.0, h_factor=1.4, v_inf=V_INF, tilt_deg=23.0
    )
    print(f"bot={bot}  top={top}  r={radius:.4f}")

    drone_pts = generate_cylindrical_sampling_coordinates(
        bot, top, radius, z_clearance=0.1, side_points=270, top_points=30
    )
    print(f"drone points: {drone_pts.shape[0]}")

    mesh = generate_momentum_integration_mesh(bot, top, radius,
                                              total_points=100, cap_top=True)
    mesh = mesh.compute_normals(cell_normals=True, point_normals=False,
                                consistent_normals=True, auto_orient_normals=True)

    pl = pv.Plotter()
    pl.add_mesh(mesh, show_edges=True, opacity=0.5, color="white")
    size = radius * 15
    pl.add_mesh(pv.Plane(center=(bot[0], bot[1], 0), direction=(0,0,1),
                         i_size=size, j_size=size), color="gray", opacity=0.2)
    arrows = mesh.cell_centers().glyph(orient="Normals", scale=False,
                                       factor=radius*0.15)
    pl.add_mesh(arrows, color="red")
    pl.add_axes()
    pl.show()
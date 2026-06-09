from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

def oblique_cylinder_geometry(stl_mesh, r_factor, h_factor, v_inf,
                               tilt_deg=23.0, clearance=0.1):
    V = np.asarray(stl_mesh.vertices)

    # downstream unit vector (horizontal only, for tilt)
    v = np.asarray(v_inf, float)
    s = np.array([v[0], v[1], 0.0])
    n = np.linalg.norm(s)
    if n < 1e-12:
        raise ValueError("v_inf has no horizontal component.")
    s = s / n

    # smallest enclosing circle radius from convex hull of footprint
    xy = np.unique(V[:, :2], axis=0)
    hull_pts = xy[ConvexHull(xy).vertices]
    footprint_circumradius = 0.5 * cdist(hull_pts, hull_pts).max()

    R = r_factor * footprint_circumradius

    # cylinder height and vertical extents
    z_lo  = V[:, 2].min()
    z_hi  = V[:, 2].max()
    z_mid = 0.5 * (z_lo + z_hi)
    H     = h_factor * (z_hi - z_lo)

    # horizontal center of bounding box
    P0 = np.array([0.5 * (V[:, 0].min() + V[:, 0].max()),
                   0.5 * (V[:, 1].min() + V[:, 1].max())])

    # tilt: axis leans downstream with height
    shift = np.tan(np.radians(tilt_deg))

    def ring_center(z):
        return np.append(P0 + shift * (z - z_mid) * s[:2], z)

    z_bottom = z_lo + clearance
    z_top    = z_bottom + H

    return ring_center(z_bottom), ring_center(z_top), R
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Geometry helpers
# ----------------------------
def rotmat(theta_deg):
    """2D rotation matrix for angle in degrees."""
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s],
                     [s,  c]])

def rectangle_vertices(center, theta_deg, s, a, h):
    """
    Rotated rectangle in the lab frame.

    Local (flow-aligned) coordinates:
        u in [-s, a]
        v in [-h, h]

    center : (x0, y0) is the local origin u=v=0 (e.g. obstacle center)
    """
    x0, y0 = center
    R = rotmat(theta_deg)

    # Local corners in counterclockwise order
    corners_local = np.array([
        [-s, -h],
        [ a, -h],
        [ a,  h],
        [-s,  h],
    ])

    corners_global = corners_local @ R.T + np.array([x0, y0])
    return corners_global

def solve_s_for_left_touch(x_obj, x_left_wall, theta_deg, h):
    """
    Choose s so the rotated rectangle touches the left wall at x = x_left_wall.

    For theta < 0 (downward tilt), the leftmost x occurs at u = -s and v = -h.
    This formula works for any theta sign using abs(sin(theta)).
    """
    th = np.deg2rad(theta_deg)
    c = np.cos(th)
    sh = np.abs(np.sin(th))

    s = (x_obj - x_left_wall - h * sh) / c
    return s

def solve_a_for_right_limit(x_obj, x_right_limit, theta_deg, h):
    """
    Choose a so the rotated rectangle stays inside the right boundary x = x_right_limit.
    """
    th = np.deg2rad(theta_deg)
    c = np.cos(th)
    sh = np.abs(np.sin(th))

    a = (x_right_limit - x_obj - h * sh) / c
    return a

def inside_sampling_region(x, y, center, theta_deg, s, a, h,
                           excl_center, excl_radius):
    """
    Boolean mask for points inside the rotated rectangle and outside the circular exclusion zone.
    """
    x0, y0 = center
    cx, cy = excl_center

    th = np.deg2rad(theta_deg)
    c, sn = np.cos(th), np.sin(th)

    # Global -> local (flow-aligned) coordinates
    dx = x - x0
    dy = y - y0
    u =  dx * c + dy * sn
    v = -dx * sn + dy * c

    in_rect = (-s <= u) & (u <= a) & (-h <= v) & (v <= h)
    in_excl = (x - cx) ** 2 + (y - cy) ** 2 <= excl_radius ** 2

    return in_rect & (~in_excl)

def sample_points(n, xlim, ylim, center, theta_deg, s, a, h,
                  excl_center, excl_radius, seed=0):
    """
    Rejection sampling inside the region.
    """
    rng = np.random.default_rng(seed)
    pts = []

    # Bounding box rejection sampling
    while len(pts) < n:
        xs = rng.uniform(xlim[0], xlim[1], size=5 * n)
        ys = rng.uniform(ylim[0], ylim[1], size=5 * n)
        mask = inside_sampling_region(xs, ys, center, theta_deg, s, a, h,
                                      excl_center, excl_radius)
        new_pts = np.column_stack([xs[mask], ys[mask]])
        pts.extend(new_pts.tolist())

    return np.array(pts[:n])

# ----------------------------
# Example setup
# ----------------------------
theta_deg = -14.5

# CFD domain limits
x_left_wall  = 0.0
x_right_wall = 12.0
y_bottom     = -4.0
y_top        = 4.0

# Obstacle / exclusion center
obj_center = (2.0, 0.0)

# Exclusion radius around the body
# Replace this with your body half-diagonal + safety margin if needed
excl_radius = 0.35

# Choose h first, then compute s so the left boundary touches the wall
h = 1.0
s = solve_s_for_left_touch(x_obj=obj_center[0],
                           x_left_wall=x_left_wall,
                           theta_deg=theta_deg,
                           h=h)

# Choose a so the right side remains inside the CFD domain
a = solve_a_for_right_limit(x_obj=obj_center[0],
                            x_right_limit=x_right_wall,
                            theta_deg=theta_deg,
                            h=h)

if s <= 0:
    raise ValueError(f"Infeasible geometry: s = {s:.4f}. Reduce h or move the obstacle rightward.")
if a <= 0:
    raise ValueError(f"Infeasible geometry: a = {a:.4f}. Reduce h or move the obstacle leftward.")

# Build rectangle boundary
verts = rectangle_vertices(obj_center, theta_deg, s, a, h)

# ----------------------------
# Sample points
# ----------------------------
pts = sample_points(
    n=500,
    xlim=(x_left_wall, x_right_wall),
    ylim=(y_bottom, y_top),
    center=obj_center,
    theta_deg=theta_deg,
    s=s, a=a, h=h,
    excl_center=obj_center,
    excl_radius=excl_radius,
    seed=42
)

print(f"s = {s:.4f}")
print(f"a = {a:.4f}")
print(f"h = {h:.4f}")
print(f"Number of samples = {len(pts)}")

# ----------------------------
# Plot
# ----------------------------
fig, ax = plt.subplots(figsize=(12, 5))

# CFD domain
ax.plot([x_left_wall, x_right_wall, x_right_wall, x_left_wall, x_left_wall],
        [y_bottom, y_bottom, y_top, y_top, y_bottom], 'k-', lw=2)

# Sampling rectangle
poly = np.vstack([verts, verts[0]])
ax.plot(poly[:, 0], poly[:, 1], 'b-', lw=2)

# Exclusion zone
circle = plt.Circle(obj_center, excl_radius, fill=False, ls='--', lw=2, color='red')
ax.add_patch(circle)

# Samples
ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.5)

ax.set_aspect('equal', adjustable='box')
ax.set_xlim(x_left_wall - 0.5, x_right_wall + 0.5)
ax.set_ylim(y_bottom - 0.5, y_top + 0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Tilted sampling region with exclusion zone')
plt.show()
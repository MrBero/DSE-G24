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
    """Calculates global (x, y) coordinates for the corners."""
    x0, y0 = center
    R = rotmat(theta_deg)
    corners_local = np.array([
        [-s, -h],
        [ a, -h],
        [ a,  h],
        [-s,  h],
    ])
    corners_global = corners_local @ R.T + np.array([x0, y0])
    return corners_global

def inside_sampling_region(x, y, center, theta_deg, s, a, h, excl_center, excl_radius):
    """Boolean mask for points inside rectangle and outside exclusion zone."""
    x0, y0 = center
    cx, cy = excl_center
    th = np.deg2rad(theta_deg)
    c, sn = np.cos(th), np.sin(th)
    dx, dy = x - x0, y - y0
    u =  dx * c + dy * sn
    v = -dx * sn + dy * c
    in_rect = (-s <= u) & (u <= a) & (-h <= v) & (v <= h)
    in_excl = (x - cx) ** 2 + (y - cy) ** 2 <= excl_radius ** 2
    return in_rect & (~in_excl)

def sample_points(n, xlim, ylim, center, theta_deg, s, a, h, excl_center, excl_radius, seed=42):
    """Rejection sampling for random interior points."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        xs = rng.uniform(xlim[0], xlim[1], size=5 * n)
        ys = rng.uniform(ylim[0], ylim[1], size=5 * n)
        mask = inside_sampling_region(xs, ys, center, theta_deg, s, a, h, excl_center, excl_radius)
        new_pts = np.column_stack([xs[mask], ys[mask]])
        pts.extend(new_pts.tolist())
    return np.array(pts[:n])

# ----------------------------
# User Setup & Variable Control
# ----------------------------
x_left_wall, x_right_wall = 0.0, 800
y_bottom, y_top = -250.0, 250
obj_center = (107.5, 0) 
theta_deg = -14.5 
excl_radius = 12.5 
excl_diam = excl_radius * 2

# Adjust 's' to change extension behind object
s = excl_diam * 2           
a = excl_diam * 3          
h = excl_diam * 2 

num_left = 50      # 'n'
num_right = 30     # 'k'
num_interior = 20 # 'X'

# ----------------------------
# Point Generation Logic
# ----------------------------
verts = rectangle_vertices(obj_center, theta_deg, s, a, h)

left_boundary_pts = np.linspace(verts[0], verts[3], num_left)
right_boundary_pts = np.linspace(verts[1], verts[2], num_right)
interior_pts = sample_points(num_interior, (x_left_wall, x_right_wall), (y_bottom, y_top), 
                             obj_center, theta_deg, s, a, h, obj_center, excl_radius)

# Combine all points and randomize their order
all_points = np.vstack([left_boundary_pts, right_boundary_pts, interior_pts])
np.random.seed(42)
np.random.shuffle(all_points)

# ----------------------------
# Output Coordinates
# ----------------------------
print(f"--- Global Corner Coordinates ---")
for label, v in zip(["Bottom-Left", "Bottom-Right", "Top-Right", "Top-Left"], verts):
    print(f"{label}: x={v[0]:.4f}, y={v[1]:.4f}")

print(f"\n--- Sample Point Coordinates (Total: {len(all_points)}) ---")
# Printing first 20 for brevity; remove [:20] to see all
for i, pt in enumerate(all_points[:num_interior]):
    print(f"Point {i+1}: x={pt[0]:.6f}, y={pt[1]:.6f}")

# ----------------------------
# Visualization
# ----------------------------
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot([x_left_wall, x_right_wall, x_right_wall, x_left_wall, x_left_wall],
        [y_bottom, y_bottom, y_top, y_top, y_bottom], 'k-', lw=2)

poly = np.vstack([verts, verts[0]])
ax.plot(poly[:, 0], poly[:, 1], 'b-', lw=1, alpha=0.5)

# Plotting randomized set
ax.scatter(all_points[:, 0], all_points[:, 1], s=10, alpha=0.6, label='Randomized Samples')

circle = plt.Circle(obj_center, excl_radius, fill=False, ls='--', lw=2, color='red')
ax.add_patch(circle)

ax.set_aspect('equal', adjustable='box')
ax.set_title('Randomized Sampling Points (Boundary + Interior)')
ax.legend()
plt.show()
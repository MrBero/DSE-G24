import numpy as np
import pandas as pd
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
x_left_wall, x_right_wall = 0.0, 800.0
y_bottom, y_top = -250.0, 250.0
obj_center = (107.5, 0) 
theta_deg = -14.5 
excl_radius = 12.5 
excl_diam = excl_radius * 2

# Adjust 's' to change extension behind object
s = excl_diam * 2           
a = excl_diam * 3          
h = excl_diam * 2 

num_left = 50      # 'n' points on left boundary
num_right = 30     # 'k' points on right boundary
num_interior = 20  # 'X' points interior

# ----------------------------
# Point Generation Logic
# ----------------------------
verts = rectangle_vertices(obj_center, theta_deg, s, a, h)

left_boundary_pts = np.linspace(verts[0], verts[3], num_left)
right_boundary_pts = np.linspace(verts[1], verts[2], num_right)
interior_pts = sample_points(num_interior, (x_left_wall, x_right_wall), (y_bottom, y_top), 
                             obj_center, theta_deg, s, a, h, obj_center, excl_radius)

# Prepare data for DataFrame (adding labels for clarity)
left_df = pd.DataFrame(left_boundary_pts, columns=['x', 'y'])
left_df['type'] = 'left_boundary'

right_df = pd.DataFrame(right_boundary_pts, columns=['x', 'y'])
right_df['type'] = 'right_boundary'

interior_df = pd.DataFrame(interior_pts, columns=['x', 'y'])
interior_df['type'] = 'interior'

# Combine and Shuffle
all_points_df = pd.concat([left_df, right_df, interior_df], ignore_index=True)
all_points_df = all_points_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ----------------------------
# Export and Output
# ----------------------------
csv_filename = 'sampling_coordinates.csv'
all_points_df.to_csv(csv_filename, index=False)
print(f"Coordinates exported to {csv_filename}")

# Print Corner Coordinates
print("\n--- Global Corner Coordinates ---")
for label, v in zip(["Bottom-Left", "Bottom-Right", "Top-Right", "Top-Left"], verts):
    print(f"{label}: x={v[0]:.4f}, y={v[1]:.4f}")

# ----------------------------
# Visualization
# ----------------------------
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot([x_left_wall, x_right_wall, x_right_wall, x_left_wall, x_left_wall],
        [y_bottom, y_bottom, y_top, y_top, y_bottom], 'k-', lw=2)

poly = np.vstack([verts, verts[0]])
ax.plot(poly[:, 0], poly[:, 1], 'b-', lw=1, alpha=0.5)

# Color-coded scatter plot using the DataFrame
colors = {'left_boundary': 'green', 'right_boundary': 'orange', 'interior': 'blue'}
for ptype, group in all_points_df.groupby('type'):
    ax.scatter(group['x'], group['y'], s=15, alpha=0.7, color=colors[ptype], label=ptype)

circle = plt.Circle(obj_center, excl_radius, fill=False, ls='--', lw=2, color='red')
ax.add_patch(circle)

ax.set_aspect('equal', adjustable='box')
ax.set_title('Randomized Sampling Points (Exported to CSV)')
ax.legend()
plt.show()
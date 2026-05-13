# sampling_region.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hashlib

# ----------------------------
# Geometry helpers
# ----------------------------
def rotmat(theta_deg):
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s],
                     [s,  c]])

def rectangle_vertices(center, theta_deg, s, a, h):
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

def sample_points(n, xlim, ylim, center, theta_deg, s, a, h, excl_center, excl_radius, seed):
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
# Seed Generation
# ----------------------------
my_string = "PEACH_VIBE"
hash_digest = hashlib.sha256(my_string.encode('utf-8')).hexdigest()
seed_number = int(hash_digest, 16) % (2**32)

print(f"Using string-derived seed: {seed_number}")

# ----------------------------
# User Setup & Variable Control
# ----------------------------
x_left_wall, x_right_wall = 0.0, 800.0
y_bottom, y_top = -250.0, 250.0
obj_center = (107.5, 0) 
theta_deg = -14.5 
excl_radius = 12.5 
excl_diam = excl_radius * 2

s, a, h = excl_diam * 3, excl_diam * 9, excl_diam * 2 

num_left, num_right, num_interior = 50, 30, 20

# ----------------------------
# Point Generation Logic
# ----------------------------
verts = rectangle_vertices(obj_center, theta_deg, s, a, h)

left_boundary_pts = np.linspace(verts[0], verts[3], num_left)
right_boundary_pts = np.linspace(verts[1], verts[2], num_right)

# FIX: Passed seed_number here to control interior randomness
interior_pts = sample_points(num_interior, (x_left_wall, x_right_wall), (y_bottom, y_top), 
                             obj_center, theta_deg, s, a, h, obj_center, excl_radius, seed=seed_number)

left_df = pd.DataFrame(left_boundary_pts, columns=['x', 'y'])
left_df['type'] = 'left_boundary'
right_df = pd.DataFrame(right_boundary_pts, columns=['x', 'y'])
right_df['type'] = 'right_boundary'

interior_df = pd.DataFrame(interior_pts, columns=['x', 'y'])
interior_df['type'] = 'interior'

all_points_df = pd.concat([left_df, right_df, interior_df], ignore_index=True)
# Shuffling also uses the seed for consistency
all_points_df = all_points_df.sample(frac=1, random_state=seed_number).reset_index(drop=True)

# ----------------------------
# Visualization
# ----------------------------
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot([x_left_wall, x_right_wall, x_right_wall, x_left_wall, x_left_wall],
        [y_bottom, y_bottom, y_top, y_top, y_bottom], 'k-', lw=2)

poly = np.vstack([verts, verts[0]])
ax.plot(poly[:, 0], poly[:, 1], 'b-', lw=1, alpha=0.5)

colors = {'left_boundary': 'green', 'right_boundary': 'orange', 'interior': 'blue'}
for ptype, group in all_points_df.groupby('type'):
    ax.scatter(group['x'], group['y'], s=15, alpha=0.7, color=colors[ptype], label=ptype)

circle = plt.Circle(obj_center, excl_radius, fill=False, ls='--', lw=2, color='red')
ax.add_patch(circle)

ax.set_aspect('equal', adjustable='box')
ax.set_title(f'Sampling Points (Seed String: {my_string})')

# FIX: Moved legend to top right as requested
ax.legend(loc='upper right')

plt.show()
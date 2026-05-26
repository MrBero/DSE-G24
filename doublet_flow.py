"""
Potential Flow Around a Sphere
================================
Superposition of:
  1. Uniform flow:  phi_inf = U * x
  2. Doublet:       phi_d   = -(kappa / 4pi) * (x / r^3),  kappa = 2*pi*U*R^3

Exact velocity field (Cartesian):
  u = U [1 - R^3 (2x^2 - y^2 - z^2) / (2r^5)]
  v = -3/2 * U * R^3 * (x*y) / r^5
  w = -3/2 * U * R^3 * (x*z) / r^5

Points inside r <= R are masked (inside the body).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Parameters ────────────────────────────────────────────────────────────────
U = 1.0        # free-stream velocity (m/s, x-direction)
R = 0.4        # sphere radius

# ── Grid (exclude a cube around the origin to avoid r=0 singularity) ─────────
N = 14         # points per axis
coords = np.linspace(-1.5, 1.5, N)
x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')

r = np.sqrt(x**2 + y**2 + z**2)
r = np.where(r < 1e-6, 1e-6, r)   # guard against division by zero

# ── Velocity components (doublet superposed on uniform flow) ──────────────────
C = 1.5 * U * R**3                 # shared prefactor = 3/2 * U * R^3

ux = U * (1.0 - R**3 * (2*x**2 - y**2 - z**2) / (2 * r**5))
uy = -C * (x * y) / r**5
uz = -C * (x * z) / r**5

# ── Mask interior of sphere ───────────────────────────────────────────────────
inside = r <= R
ux[inside] = np.nan
uy[inside] = np.nan
uz[inside] = np.nan

# ── Normalise arrow lengths for display ──────────────────────────────────────
mag = np.sqrt(ux**2 + uy**2 + uz**2)
mag = np.where(np.isnan(mag) | (mag < 1e-8), 1.0, mag)
scale = 0.18   # arrow display length

ux_n = np.nan_to_num(ux / mag) * scale
uy_n = np.nan_to_num(uy / mag) * scale
uz_n = np.nan_to_num(uz / mag) * scale

# ── Colour arrows by speed magnitude ─────────────────────────────────────────
speed = np.sqrt(ux**2 + uy**2 + uz**2)
speed_flat = speed.ravel()
valid = ~np.isnan(speed_flat)
s_min, s_max = np.nanmin(speed), np.nanmax(speed)
norm_speed = np.nan_to_num((speed - s_min) / (s_max - s_min + 1e-12))
colors = plt.cm.plasma(norm_speed.ravel())

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

# Vector field — only exterior points
mask = ~inside.ravel()
ax.quiver(
    x.ravel()[mask], y.ravel()[mask], z.ravel()[mask],
    ux_n.ravel()[mask], uy_n.ravel()[mask], uz_n.ravel()[mask],
    colors=colors[mask],
    linewidth=0.8, arrow_length_ratio=0.35
)

# Sphere surface
u_s = np.linspace(0, 2*np.pi, 60)
v_s = np.linspace(0, np.pi, 40)
xs = R * np.outer(np.cos(u_s), np.sin(v_s))
ys = R * np.outer(np.sin(u_s), np.sin(v_s))
zs = R * np.outer(np.ones_like(u_s), np.cos(v_s))
ax.plot_surface(xs, ys, zs, color='steelblue', alpha=0.35,
                linewidth=0, antialiased=True)

# Stagnation-point markers (front & rear, on x-axis at r=R)
ax.scatter([ R, -R], [0, 0], [0, 0],
           color='red', s=60, zorder=5, label='Stagnation points')

# ── Labels & style ────────────────────────────────────────────────────────────
ax.set_xlabel('x', labelpad=8)
ax.set_ylabel('y', labelpad=8)
ax.set_zlabel('z', labelpad=8)
ax.set_title(f'Potential Flow Around a Sphere  (R = {R}, U = {U})',
             fontsize=13, pad=14)
ax.legend(loc='upper left', fontsize=9)
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=22, azim=-55)

# Colourbar proxy
sm = plt.cm.ScalarMappable(cmap='plasma',
                            norm=plt.Normalize(vmin=s_min, vmax=s_max))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08)
cbar.set_label('|u| / U∞', fontsize=10)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/potential_flow_sphere.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("Saved → potential_flow_sphere.png")


# ── 2-D slice for cleaner visualisation (z = 0 plane) ────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 7))

N2  = 40
lim = 1.5
xg, yg = np.meshgrid(np.linspace(-lim, lim, N2),
                      np.linspace(-lim, lim, N2))
rg  = np.sqrt(xg**2 + yg**2)
rg  = np.where(rg < 1e-6, 1e-6, rg)

ux2 = U * (1.0 - R**3 * (2*xg**2 - yg**2) / (2 * rg**5))
uy2 = -C * (xg * yg) / rg**5
ux2[rg <= R] = np.nan
uy2[rg <= R] = np.nan

speed2 = np.sqrt(ux2**2 + uy2**2)

# Streamlines
ax2.streamplot(xg, yg, ux2, uy2, density=1.8,
               color=speed2, cmap='viridis',
               linewidth=1.2, arrowsize=1.2)

# Sphere
theta_circ = np.linspace(0, 2*np.pi, 200)
ax2.fill(R*np.cos(theta_circ), R*np.sin(theta_circ),
         color='steelblue', alpha=0.7, zorder=3)
ax2.plot(R*np.cos(theta_circ), R*np.sin(theta_circ),
         'k-', lw=1.5, zorder=4)

# Stagnation points
ax2.plot([ R, -R], [0, 0], 'ro', ms=7, zorder=5, label='Stagnation points')

sm2 = plt.cm.ScalarMappable(cmap='viridis',
       norm=plt.Normalize(vmin=np.nanmin(speed2), vmax=np.nanmax(speed2)))
sm2.set_array([])
cbar2 = fig2.colorbar(sm2, ax=ax2)
cbar2.set_label('|u| / U∞', fontsize=11)

ax2.set_xlabel('x / R', fontsize=11)
ax2.set_ylabel('y / R', fontsize=11)
ax2.set_title('Potential Flow — z = 0 slice  (streamlines + speed)', fontsize=13)
ax2.set_aspect('equal')
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/potential_flow_slice.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("Saved → potential_flow_slice.png")
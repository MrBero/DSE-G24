import numpy as np
import pandas as pd
import pyvista
from INTERP import interpolation
from momentum import momentum_closed, momentum_open


PKL_PATH = r"INTERP\csv_with_everything.pkl"
MIDPOINT = np.array([
	6524.591 / 1000,
	1.213 * 10**5 / 10**3,
	52.6 / 2
])
N_POINTS = 4
SHARPNESS = 2

def sample_cylinder_uniform(
	p1: np.ndarray,
	p2: np.ndarray,
	R: float,
	n: int,
	top: bool = False,
	bottom: bool = False,
) -> np.ndarray:
	"""Sample staggered uniform points on the lateral surface of an oblique
	cylinder defined by p1 (center of bottom base) and p2 (center of top
	base), with radius R. Both bases are assumed parallel to the XY plane.

	Points are distributed so that points-per-unit-area is equal across
	the lateral surface and any enabled caps. n is the total point budget.

	Args:
		p1:     Center of the bottom base, shape (3,).
		p2:     Center of the top base, shape (3,).
		R:      Radius of both circular bases.
		n:      Total number of points across all sampled surfaces.
		top:    If True, also sample points on the top cap disc.
		bottom: If True, also sample points on the bottom cap disc.

	Returns:
		Array of shape (N, 3) with all sampled points in world space.
	"""
	p1 = np.asarray(p1, dtype=float)
	p2 = np.asarray(p2, dtype=float)

	H = p2[2] - p1[2]
	if H == 0.0:
		raise ValueError("p1 and p2 must differ in z (both bases are parallel to XY plane).")

	axis = p2 - p1
	L = np.linalg.norm(axis)  # slant/axial length

	# --- Area of each surface ---
	A_lateral = 2 * np.pi * R * L
	A_cap = np.pi * R ** 2
	n_caps = int(bottom) + int(top)
	A_total = A_lateral + n_caps * A_cap

	# --- Allocate points proportionally to area ---
	density = n / A_total  # points per unit area
	n_lateral = max(1, round(density * A_lateral))
	n_cap = max(1, round(density * A_cap)) if n_caps > 0 else 0

	# --- Lateral surface ---
	n_cols = max(1, round(np.sqrt(n_lateral * 2 * np.pi * R / L)))
	n_rows = max(1, round(n_lateral / n_cols))

	idx = np.arange(n_rows * n_cols)
	row_idx = idx // n_cols
	col_idx = idx % n_cols

	# Staggered theta per row
	theta_shift = (row_idx % 2) * (np.pi / n_cols)
	theta = (2 * np.pi * col_idx / n_cols) + theta_shift

	# Center rows within [0, 1] -- avoids collapse when n_rows=1
	t = (row_idx + 0.5) / n_rows

	cx = p1[0] + t * (p2[0] - p1[0])
	cy = p1[1] + t * (p2[1] - p1[1])
	z  = p1[2] + t * H

	points = [np.c_[cx + R * np.cos(theta), cy + R * np.sin(theta), z]]

	# --- Cap helper ---
	def sample_disc(center: np.ndarray, n_pts: int) -> np.ndarray:
		"""Sunflower/Fibonacci spiral for uniform disc coverage."""
		golden = np.pi * (3.0 - np.sqrt(5.0))  # ~137.5 degrees
		k = np.arange(n_pts)
		r = R * np.sqrt((k + 0.5) / n_pts)
		a = golden * k
		return np.c_[
			center[0] + r * np.cos(a),
			center[1] + r * np.sin(a),
			np.full(n_pts, center[2]),
		]

	if bottom:
		points.append(sample_disc(p1, n_cap))
	if top:
		points.append(sample_disc(p2, n_cap))

	return np.vstack(points)


region_points = sample_cylinder_uniform(
	p1=[MIDPOINT[0], MIDPOINT[1], 0],
	p2=[MIDPOINT[0], MIDPOINT[1], 90],
	R=40,
	n=100_000,
	top=True,
	bottom=False,
)

# print('x_min, ', np.min(region_points[:,0]))
# print('y_min ', np.min(region_points[:,1]))
# print('z_min ', np.min(region_points[:,2]))


cloud = pyvista.PolyData(region_points)

# print('build interp')
df = pd.read_pickle(PKL_PATH)
sample = interpolation.build_cfd_sampler(df, n_points=N_POINTS, sharpness=SHARPNESS)

# print(sample([[-6524.591 / 1000, 1.213 * 10**5 / 10**3, 0]]))

# print('interp points')
out = sample(region_points)
v = out[:, :3]
p = out[:, 3]

# print(np.isnan(v).any())
# print(np.isnan(p).any())

# print('interp done')

print('compute force')
# F, mesh = momentum_closed.surface_force(
# 	points=region_points,
# 	center=MIDPOINT,
# 	velocity=v,
# 	pressure=p,
# 	rho=1.225,
# )

F, mesh, _ = momentum_open.surface_force_bpa(
	points=region_points,
	center=MIDPOINT,
	velocity=v,
	pressure=p,
	rho=1.225,
	L=10
)


print(F)

arrows = mesh.glyph(orient="Normals", scale=False, factor=5)

import numpy as np

def add_reference_axes(plotter, mesh, n_ticks=5, axis_extension=0.1):
	"""
	Draw XYZ axes from min to max of the mesh bounds, with tick labels.
	axis_extension: fraction to extend beyond the mesh bounds (10% default)
	"""
	bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
	axes = [
		{"name": "X", "color": "red",   "min": bounds[0], "max": bounds[1], "dir": (1,0,0), "perp": (0,1,0)},
		{"name": "Y", "color": "green", "min": bounds[2], "max": bounds[3], "dir": (0,1,0), "perp": (1,0,0)},
		{"name": "Z", "color": "blue",  "min": bounds[4], "max": bounds[5], "dir": (0,0,1), "perp": (1,0,0)},
	]

	for ax in axes:
		lo, hi = ax["min"], ax["max"]
		span = hi - lo
		ext = span * axis_extension
		d = ax["dir"]
		p = ax["perp"]

		# axis line from min to max (with small extension)
		start = (d[0]*(lo - ext), d[1]*(lo - ext), d[2]*(lo - ext))
		end   = (d[0]*(hi + ext), d[1]*(hi + ext), d[2]*(hi + ext))
		plotter.add_mesh(pyvista.Line(start, end), color=ax["color"], line_width=2)

		# ticks + labels
		for val in np.linspace(lo, hi, n_ticks):
			pos       = [d[i] * val for i in range(3)]
			tick_end  = [pos[i] + p[i] * span * 0.02 for i in range(3)]  # small tick mark

			plotter.add_mesh(pyvista.Line(pos, tick_end), color=ax["color"], line_width=2)
			plotter.add_point_labels(
				[pos],
				[f"{val:.1f}"],
				font_size=10,
				text_color=ax["color"],
				show_points=False,
				always_visible=True,
			)

		# axis name label at the end
		name_pos = [d[i] * (hi + ext * 2) for i in range(3)]
		plotter.add_point_labels(
			[name_pos],
			[ax["name"]],
			font_size=14,
			text_color=ax["color"],
			show_points=False,
			always_visible=True,
		)

# usage
plotter = pyvista.Plotter()
plotter.add_mesh(mesh, scalars="pressure")
plotter.add_mesh(arrows, color="red")
add_reference_axes(plotter, mesh, n_ticks=6)
plotter.show()
plotter.close()

del plotter

import os
del cloud
os._exit(0)

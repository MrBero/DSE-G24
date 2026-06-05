import numpy as np

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


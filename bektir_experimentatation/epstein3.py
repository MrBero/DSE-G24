"""
Oblique cylinder mesh: both caps are horizontal (z = const),
but the lateral wall leans — the top cap centre is offset in XY
from the bottom cap centre, so the axis is tilted.
"""

import numpy as np
import pyvista as pv


import numpy as np
import pyvista as pv

def make_oblique_cylinder_mesh(
	center_bot: np.ndarray,
	center_top: np.ndarray,
	radius: float,
	total_points: int = 100_000,
	cap_bottom: bool = False,
	cap_top: bool = False,
) -> pv.PolyData:
	"""
	Build an oblique cylinder: horizontal flat caps, slanted triangulated wall.
	"""
	center_bot = np.asarray(center_bot, dtype=float)
	center_top = np.asarray(center_top, dtype=float)

	cx_b, cy_b, z_bottom = center_bot
	cx_t, cy_t, z_top    = center_top

	axis_vec    = center_top - center_bot
	axis_length = float(np.linalg.norm(axis_vec))

	n_caps       = int(cap_top) + int(cap_bottom)
	lateral_area = 2 * np.pi * radius * axis_length
	cap_area     = np.pi * radius ** 2
	total_area   = lateral_area + n_caps * cap_area

	s = np.sqrt(total_area / total_points)

	T           = max(4, round(2 * np.pi * radius / s))
	Z           = max(1, round(axis_length / s))
	cap_r_res   = max(1, round(radius / s))

	thetas = np.linspace(0, 2 * np.pi, T, endpoint=False)
	dtheta = 2 * np.pi / T
	
	# Apply half-step shift to odd rings to stagger points
	shifts = (np.arange(Z + 1) % 2) * (dtheta / 2.0)
	thetas_2d = thetas[None, :] + shifts[:, None]

	cos_t_2d = np.cos(thetas_2d)
	sin_t_2d = np.sin(thetas_2d)

	# ------------------------------------------------------------------
	# 1. Lateral surface vertices
	# ------------------------------------------------------------------
	alphas   = np.linspace(0.0, 1.0, Z + 1)
	cx_rings = cx_b + alphas * (cx_t - cx_b)
	cy_rings = cy_b + alphas * (cy_t - cy_b)
	z_rings  = z_bottom + alphas * (z_top - z_bottom)

	lat_x = cx_rings[:, None] + radius * cos_t_2d
	lat_y = cy_rings[:, None] + radius * sin_t_2d
	lat_z = np.tile(z_rings[:, None], (1, T))

	lat_pts = np.stack(
		[lat_x.ravel(), lat_y.ravel(), lat_z.ravel()], axis=1
	)
	n_lat = lat_pts.shape[0]

	# ------------------------------------------------------------------
	# 2. Cap interior vertices
	# ------------------------------------------------------------------
	cap_radii = radius * np.arange(cap_r_res - 1, 0, -1) / cap_r_res
	
	def make_cap_pts(cx, cy, z, shift):
		pts = []
		if cap_r_res > 1:
			cap_thetas = thetas + shift
			c_t = np.cos(cap_thetas)
			s_t = np.sin(cap_thetas)
			for r_c in cap_radii:
				xs = cx + r_c * c_t
				ys = cy + r_c * s_t
				zs = np.full(T, z)
				pts.append(np.stack([xs, ys, zs], axis=1))
		pts.append(np.array([[cx, cy, z]]))
		return np.vstack(pts)

	all_pts_list = [lat_pts]
	bot_cap_offset = None
	top_cap_offset = None

	if cap_bottom:
		bot_cap_pts    = make_cap_pts(cx_b, cy_b, z_bottom, shifts[0])
		bot_cap_offset = n_lat
		all_pts_list.append(bot_cap_pts)

	if cap_top:
		top_cap_pts    = make_cap_pts(cx_t, cy_t, z_top, shifts[-1])
		top_cap_offset = n_lat + (bot_cap_pts.shape[0] if cap_bottom else 0)
		all_pts_list.append(top_cap_pts)

	all_pts = np.vstack(all_pts_list)

	# ------------------------------------------------------------------
	# 3. Index helpers
	# ------------------------------------------------------------------
	def lat_idx(i, j):
		return i * T + (j % T)

	def cap_ring_idx(offset, ring, j):
		return offset + ring * T + (j % T)

	def cap_centre_idx(offset):
		return offset + (cap_r_res - 1) * T

	# ------------------------------------------------------------------
	# 4. Faces
	# ------------------------------------------------------------------
	faces = []

	# Lateral staggered triangles
	for i in range(Z):
		if i % 2 == 0:
			# Lower ring unshifted, upper ring shifted (+0.5 dtheta)
			for j in range(T):
				j_next = (j + 1) % T
				a = lat_idx(i, j)
				b = lat_idx(i, j_next)
				c = lat_idx(i + 1, j)
				d = lat_idx(i + 1, j_next)
				faces.extend([3, a, b, c])
				faces.extend([3, c, b, d])
		else:
			# Lower ring shifted (+0.5 dtheta), upper ring unshifted
			for j in range(T):
				j_next = (j + 1) % T
				a = lat_idx(i, j)
				b = lat_idx(i, j_next)
				c = lat_idx(i + 1, j)
				d = lat_idx(i + 1, j_next)
				faces.extend([3, a, b, d])
				faces.extend([3, c, a, d])

	# Cap faces completely triangulated
	def add_cap_faces(lat_ring_i, offset, inward):
		centre = cap_centre_idx(offset)
		if cap_r_res >= 2:
			for j in range(T):
				j_next = (j + 1) % T
				a = lat_idx(lat_ring_i, j)
				b = lat_idx(lat_ring_i, j_next)
				c = cap_ring_idx(offset, 0, j_next)
				d = cap_ring_idx(offset, 0, j)
				if inward:
					faces.extend([3, a, d, c, 3, a, c, b])
				else:
					faces.extend([3, a, b, c, 3, a, c, d])

			for ring in range(cap_r_res - 2):
				for j in range(T):
					j_next = (j + 1) % T
					a = cap_ring_idx(offset, ring,     j)
					b = cap_ring_idx(offset, ring,     j_next)
					c = cap_ring_idx(offset, ring + 1, j_next)
					d = cap_ring_idx(offset, ring + 1, j)
					if inward:
						faces.extend([3, a, d, c, 3, a, c, b])
					else:
						faces.extend([3, a, b, c, 3, a, c, d])

			inner = cap_r_res - 2
			for j in range(T):
				j_next = (j + 1) % T
				a = cap_ring_idx(offset, inner, j)
				b = cap_ring_idx(offset, inner, j_next)
				faces.extend([3, a, centre, b] if inward else [3, a, b, centre])
		else:
			for j in range(T):
				j_next = (j + 1) % T
				a = lat_idx(lat_ring_i, j)
				b = lat_idx(lat_ring_i, j_next)
				faces.extend([3, a, centre, b] if inward else [3, a, b, centre])

	if cap_bottom:
		add_cap_faces(0, bot_cap_offset, inward=True)
	if cap_top:
		add_cap_faces(Z, top_cap_offset, inward=False)

	return pv.PolyData(all_pts, np.array(faces, dtype=np.int_))


# ---------------------------------------------------------------------------
# CFD helpers (unchanged from original)
# ---------------------------------------------------------------------------

def attach_cfd_fields(mesh: pv.PolyData, sampler) -> pv.PolyData:
	out = sampler(mesh.points)
	mesh["velocity"] = out[:, :3]
	mesh["pressure"] = out[:, 3]
	return mesh


def surface_force(
	mesh: pv.PolyData,
	rho: float = 1.225,
) -> tuple[np.ndarray, pv.PolyData]:
	if "velocity" not in mesh.point_data or "pressure" not in mesh.point_data:
		raise ValueError(
			"Mesh must have 'velocity' and 'pressure' point arrays. "
			"Call attach_cfd_fields(mesh, sampler) first."
		)
	
	# 1. Compute pointwise velocity tensor (v_i * v_j) before conversion
	v_pt = mesh.point_data["velocity"]
	M_pt = v_pt[:, :, np.newaxis] * v_pt[:, np.newaxis, :]
	mesh.point_data["v_tensor"] = M_pt.reshape(-1, 9)

	# 2. Compute normals and interpolate all point arrays to cell centers
	mesh = mesh.compute_normals(
		cell_normals=True, point_normals=False,
		consistent_normals=False, auto_orient_normals=False,
	)
	mesh = mesh.point_data_to_cell_data()
	
	n = mesh.cell_data["Normals"]
	p = mesh.cell_data["pressure"]
	
	# 3. Compute centroid velocity tensor (v_avg_i * v_avg_j)
	v_cell = mesh.cell_data["velocity"]
	T_center = v_cell[:, :, np.newaxis] * v_cell[:, np.newaxis, :]
	
	# 4. Extract averaged pointwise tensor (Avg(v_node_i * v_node_j))
	T_avg = mesh.cell_data["v_tensor"].reshape(-1, 3, 3)
	
	# 5. Apply exact analytical quadratic integration over linear triangles
	M_exact = 0.25 * T_avg + 0.75 * T_center
	
	# 6. Apply force computation
	# M_exact is (N, 3, 3), n is (N, 3). Einsum resolves sum_j (M_ij * n_j)
	mesh.cell_data["pressure_force"] = -p[:, np.newaxis] * n
	mesh.cell_data["momentum_force"] = -rho * np.einsum('nij,nj->ni', M_exact, n)
	
	mesh.cell_data["total_force"] = (
		mesh.cell_data["pressure_force"] + mesh.cell_data["momentum_force"]
	)
	
	F = mesh.integrate_data().cell_data["total_force"][0]
	return F, mesh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
	Radius   = 52.6
	r_factor = 0.8

	R  = 60
	H  = 70.0
	N_POINTS  = 1
	SHARPNESS = 4

	MIDPOINT = np.array([
		6524.591 / 1000,
		1.213 * 10**5 / 10**3,
		52.6 / 2,
	])

	PKL_PATH   = r"INTERP\csv_with_everything.pkl"
	CACHE_PATH = r"inputs/cfd_sampler_cache.joblib"

	import time
	import pandas as pd
	import matplotlib.pyplot as plt
	from INTERP import interpolation

	t = time.time()
	print("starting")

	df      = pd.read_pickle(PKL_PATH)
	sampler = interpolation.build_cfd_sampler(
		df, n_points=N_POINTS, sharpness=SHARPNESS, cache_path=CACHE_PATH
	)
	print(time.time() - t, "sampler done")

	# Define coordinates before generating the mesh
	CX, CY = float(MIDPOINT[0]), float(MIDPOINT[1])
	F_REF_Y = 208647.0

	# Instantiate the surface mesh and attach CFD fields
	surface = make_oblique_cylinder_mesh(
		center_bot   = np.array([CX, CY, 0.0]),
		center_top   = np.array([CX, CY, H]),
		radius       = R,
		total_points = 300,
		cap_top      = True,
		cap_bottom   = False,
	)
	attach_cfd_fields(surface, sampler)

	# Unpack the tuple returned by surface_force
	F_total, result_mesh = surface_force(surface, rho=1.225)
	print(F_total)
	
	pl = pv.Plotter()
	pl.add_mesh(result_mesh, scalars="pressure", cmap="viridis",
				show_edges=True, opacity=0.8)
	
	# Extract cell centers using the unpacked mesh object
	centers = result_mesh.cell_centers()
	
	pl.add_arrows(centers.points, result_mesh.cell_data["Normals"], mag=5.0, color="red")
	
	pl.add_axes()
	pl.show()

	# ---- convergence study: POINT COUNT --------------------------------
	r_factors = np.linspace(0.5, 2, 10)
	fy_r = []

	for rf in r_factors:
		R_rf = Radius * rf
		m = make_oblique_cylinder_mesh(
			center_bot   = np.array([CX, CY, 0.0]),
			center_top   = np.array([CX, CY, H]),
			radius       = R_rf,
			total_points = 1_000_000,
			cap_top      = True,
			cap_bottom   = False,
		)

		attach_cfd_fields(m, sampler)
		F_n, _ = surface_force(m, rho=1.225)
		fy_r.append(F_n[1])
		print(f"r_factor={rf:.3f} R={R_rf:.2f} Fy={F_n[1]:>12.1f}")

	# Plotting block unindented
	plt.figure()
	plt.axhline(F_REF_Y, color="red", linestyle="--", label=f"Reference Fy = {F_REF_Y:,.0f} N")
	plt.plot(r_factors, fy_r, marker="o")
	plt.xlabel("r_factor")
	plt.ylabel("Fy [N]")
	plt.title("Convergence study - Fy vs r_factor")
	plt.legend()
	plt.tight_layout()
	plt.show()

	# ---- convergence study: HEIGHT ---------------------------------------
	heights = np.linspace(60, 90, 31)
	fy_h = []

	for h in heights:
		m = make_oblique_cylinder_mesh(
			center_bot   = np.array([CX, CY, 0.0]),
			center_top   = np.array([CX, CY, h]),
			radius       = R,
			total_points = 250_000,
			cap_top      = True,
			cap_bottom   = False,
		)

		attach_cfd_fields(m, sampler)
		F_n, _ = surface_force(m, rho=1.225)
		fy_h.append(F_n[1])
		print(f"height={h:.2f} Fy={F_n[1]:>12.1f}")

	plt.figure()
	plt.axhline(F_REF_Y, color="red", linestyle="--", label=f"Reference Fy = {F_REF_Y:,.0f} N")
	plt.plot(heights, fy_h, marker="o")
	plt.xlabel("Height [m]")
	plt.ylabel("Fy [N]")
	plt.title("Convergence study - Fy vs Height")
	plt.legend()
	plt.tight_layout()
	plt.show()
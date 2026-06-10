from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import numpy as np
import pyvista as pv

def generate_square_momentum_integration_mesh(
	center_bot: np.ndarray,
	center_top: np.ndarray,
	radius: float,
	total_points: int = 100_000,
	cap_bottom: bool = False,
	cap_top: bool = False,
) -> pv.PolyData:
	"""
	Build an oblique cuboid: horizontal flat square caps, slanted triangulated walls.
 
	The square cross-section has side length = radius * 2, so `radius` is the
	half-side length. Points are distributed to match roughly the requested
	total_points density across all active surfaces.
	"""
	center_bot = np.asarray(center_bot, dtype=float)
	center_top = np.asarray(center_top, dtype=float)
 
	cx_b, cy_b, z_bottom = center_bot
	cx_t, cy_t, z_top    = center_top
 
	axis_vec    = center_top - center_bot
	axis_length = float(np.linalg.norm(axis_vec))
 
	# half-side: radius*2 is the full side length
	half = radius
 
	# ------------------------------------------------------------------
	# Surface area budget to derive grid spacing s
	# ------------------------------------------------------------------
	n_caps       = int(cap_top) + int(cap_bottom)
	lateral_area = 4 * (2 * half) * axis_length   # 4 rectangular faces
	cap_area     = (2 * half) ** 2                 # square cap
	total_area   = lateral_area + n_caps * cap_area
 
	s = np.sqrt(total_area / total_points)
 
	# Segments along axis and along one edge of the square perimeter
	Z = max(1, round(axis_length / s))
	E = max(2, round((2 * half) / s))   # segments per edge
 
	# Total perimeter samples: 4 edges * E points, no duplicate corners
	# We index perimeter as 4*E points going around the square
	P = 4 * E
 
	# ------------------------------------------------------------------
	# Build perimeter offsets for one ring at unit square [-half, half]^2
	# Corner order: bottom-left -> bottom-right -> top-right -> top-left
	#   edge 0: bottom  y=-half, x from -half to +half  (E pts, skip last)
	#   edge 1: right   x=+half, y from -half to +half  (E pts, skip last)
	#   edge 2: top     y=+half, x from +half to -half  (E pts, skip last)
	#   edge 3: left    x=-half, y from +half to -half  (E pts, skip last)
	# ------------------------------------------------------------------
	def make_perimeter_offsets(shift_frac: float = 0.0):
		"""Return (dx, dy) arrays of shape (P,) around the square perimeter.
		shift_frac in [0, 1) offsets all points by that fraction of one segment."""
		t_edge = np.linspace(0.0, 1.0, E, endpoint=False)  # (E,)
		t_edge = (t_edge + shift_frac / E) % 1.0
 
		# edge 0: bottom  y=-half, x = -half + t*2*half
		dx0 = -half + t_edge * 2 * half
		dy0 = np.full(E, -half)
 
		# edge 1: right   x=+half, y = -half + t*2*half
		dx1 = np.full(E, half)
		dy1 = -half + t_edge * 2 * half
 
		# edge 2: top     y=+half, x = half - t*2*half
		dx2 = half - t_edge * 2 * half
		dy2 = np.full(E, half)
 
		# edge 3: left    x=-half, y = half - t*2*half
		dx3 = np.full(E, -half)
		dy3 = half - t_edge * 2 * half
 
		dx = np.concatenate([dx0, dx1, dx2, dx3])
		dy = np.concatenate([dy0, dy1, dy2, dy3])
		return dx, dy
 
	# Half-segment shift (analogous to dtheta/2 in the cylinder) for staggering
	half_shift = 0.5
 
	alphas   = np.linspace(0.0, 1.0, Z + 1)
	cx_rings = cx_b + alphas * (cx_t - cx_b)
	cy_rings = cy_b + alphas * (cy_t - cy_b)
	z_rings  = z_bottom + alphas * (z_top - z_bottom)
 
	# Stagger: odd rings are shifted by half a segment
	shifts = (np.arange(Z + 1) % 2) * half_shift  # (Z+1,)
 
	# ------------------------------------------------------------------
	# 1. Lateral surface vertices
	# ------------------------------------------------------------------
	lat_pts_list = []
	for i in range(Z + 1):
		dx, dy = make_perimeter_offsets(shifts[i])
		xs = cx_rings[i] + dx
		ys = cy_rings[i] + dy
		zs = np.full(P, z_rings[i])
		lat_pts_list.append(np.stack([xs, ys, zs], axis=1))
 
	lat_pts = np.vstack(lat_pts_list)   # shape: ((Z+1)*P, 3)
	n_lat   = lat_pts.shape[0]
 
	# ------------------------------------------------------------------
	# 2. Cap interior vertices (grid inside the square)
	# ------------------------------------------------------------------
	# Use a regular grid for cap interior, spacing ~s
	cap_n = max(2, round(2 * half / s))  # grid lines per axis (including edges)
 
	def make_cap_pts(cx, cy, z):
		"""Fill the interior of the square cap with a regular grid,
		excluding the perimeter points (those are the lateral ring)."""
		ts = np.linspace(-half, half, cap_n + 1)
		# Interior only: skip first and last (those live on the lateral ring)
		ts_inner = ts[1:-1]
		if ts_inner.size == 0:
			# No interior; just the centre point
			return np.array([[cx, cy, z]])
		gx, gy = np.meshgrid(ts_inner, ts_inner)
		xs = cx + gx.ravel()
		ys = cy + gy.ravel()
		zs = np.full(xs.size, z)
		# Add centre explicitly to ensure it's present
		interior = np.stack([xs, ys, zs], axis=1)
		return interior
 
	all_pts_list    = [lat_pts]
	bot_cap_offset  = None
	top_cap_offset  = None
	bot_cap_pts     = None
	top_cap_pts     = None
 
	if cap_bottom:
		bot_cap_pts    = make_cap_pts(cx_b, cy_b, z_bottom)
		bot_cap_offset = n_lat
		all_pts_list.append(bot_cap_pts)
 
	if cap_top:
		top_cap_pts    = make_cap_pts(cx_t, cy_t, z_top)
		top_cap_offset = n_lat + (bot_cap_pts.shape[0] if cap_bottom else 0)
		all_pts_list.append(top_cap_pts)
 
	all_pts = np.vstack(all_pts_list)
 
	# ------------------------------------------------------------------
	# 3. Index helpers
	# ------------------------------------------------------------------
	def lat_idx(i, j):
		"""Global index of ring i, perimeter position j (wraps)."""
		return i * P + (j % P)
 
	# ------------------------------------------------------------------
	# 4. Lateral faces (staggered triangles, same logic as cylinder)
	# ------------------------------------------------------------------
	faces = []
 
	for i in range(Z):
		if i % 2 == 0:
			for j in range(P):
				j_next = (j + 1) % P
				a = lat_idx(i,     j)
				b = lat_idx(i,     j_next)
				c = lat_idx(i + 1, j)
				d = lat_idx(i + 1, j_next)
				faces.extend([3, a, b, c])
				faces.extend([3, c, b, d])
		else:
			for j in range(P):
				j_next = (j + 1) % P
				a = lat_idx(i,     j)
				b = lat_idx(i,     j_next)
				c = lat_idx(i + 1, j)
				d = lat_idx(i + 1, j_next)
				faces.extend([3, a, b, d])
				faces.extend([3, c, a, d])
 
	# ------------------------------------------------------------------
	# 5. Cap faces
	# For each cap we use a Delaunay triangulation of the perimeter ring
	# plus interior grid points projected to 2-D, so the connectivity is
	# correct regardless of interior density.
	# ------------------------------------------------------------------
	def add_cap_faces_delaunay(lat_ring_i, cap_offset, cap_n_pts, inward):
		"""
		Triangulate the cap using a 2-D Delaunay (via pyvista/scipy) of
		the perimeter ring + interior points, then map back to global indices.
		"""
		from scipy.spatial import Delaunay
 
		# Collect 2-D positions of perimeter ring (ring lat_ring_i)
		perim_global = [lat_idx(lat_ring_i, j) for j in range(P)]
		perim_xy     = all_pts[perim_global, :2]   # (P, 2) - use x,y for 2D
 
		# Interior points
		if cap_n_pts > 0:
			interior_global = list(range(cap_offset, cap_offset + cap_n_pts))
			interior_xy     = all_pts[interior_global, :2]
			all_local_xy    = np.vstack([perim_xy, interior_xy])
			all_global_idx  = perim_global + interior_global
		else:
			all_local_xy   = perim_xy
			all_global_idx = perim_global
 
		tri = Delaunay(all_local_xy)
		for simplex in tri.simplices:
			a, b, c = all_global_idx[simplex[0]], all_global_idx[simplex[1]], all_global_idx[simplex[2]]
			if inward:
				faces.extend([3, a, c, b])
			else:
				faces.extend([3, a, b, c])
 
	if cap_bottom:
		n_bot = bot_cap_pts.shape[0]
		add_cap_faces_delaunay(0, bot_cap_offset, n_bot, inward=True)
 
	if cap_top:
		n_top = top_cap_pts.shape[0]
		add_cap_faces_delaunay(Z, top_cap_offset, n_top, inward=False)
 
	return pv.PolyData(all_pts, np.array(faces, dtype=np.int_))


def generate_momentum_integration_mesh(
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


def generate_cylindrical_sampling_coordinates(
	center_bot: np.ndarray,
	center_top: np.ndarray,
	radius: float,
	z_clearance: float = 0.1,
	side_points: int = 0,
	top_points: int = 0,
	bot_points: int = 0,
) -> np.ndarray:
	PHI          = (1 + np.sqrt(5)) / 2
	GOLDEN_ANGLE = 2 * np.pi * (1 - 1 / PHI)

	center_bot = np.asarray(center_bot, dtype=float)
	center_top = np.asarray(center_top, dtype=float)

	axis_vec  = center_top - center_bot
	axis_unit = axis_vec / np.linalg.norm(axis_vec)

	# Shift the bottom center along the axis vector so that the z component
	# of the shift equals z_clearance. This preserves the oblique angle.
	z_shift_along_axis = z_clearance / axis_unit[2]
	center_bot_shifted = center_bot + axis_unit * z_shift_along_axis

	cx_b, cy_b, z_bottom = center_bot_shifted
	cx_t, cy_t, z_top    = center_top

	all_pts_list = []

	# --- Lateral (side) surface sampling ---
	if side_points > 0:
		i      = np.arange(side_points)
		alphas = i / (side_points - 1) if side_points > 1 else np.array([0.5])
		thetas = i * GOLDEN_ANGLE

		cx = cx_b + alphas * (cx_t - cx_b)
		cy = cy_b + alphas * (cy_t - cy_b)
		z  = z_bottom + alphas * (z_top - z_bottom)

		xs = cx + radius * np.cos(thetas)
		ys = cy + radius * np.sin(thetas)

		all_pts_list.append(np.stack([xs, ys, z], axis=1))

	# --- Cap sampling ---
	def make_cap_pts(cx, cy, z, n):
		i      = np.arange(n)
		r      = radius * np.sqrt((i + 0.5) / n)
		thetas = i * GOLDEN_ANGLE

		xs = cx + r * np.cos(thetas)
		ys = cy + r * np.sin(thetas)
		zs = np.full(n, z)

		return np.stack([xs, ys, zs], axis=1)

	if bot_points > 0:
		all_pts_list.append(make_cap_pts(cx_b, cy_b, z_bottom, bot_points))

	if top_points > 0:
		all_pts_list.append(make_cap_pts(cx_t, cy_t, z_top, top_points))

	return np.vstack(all_pts_list)


def calculate_wake_cylinder_parameters(
	stl_mesh, 
	r_factor, 
	h_factor, 
	v_inf,
	tilt_deg=23.0
):
	vertices = np.asarray(stl_mesh.vertices)

	# Downstream unit vector (horizontal only, for tilt)
	v_inf_array = np.asarray(v_inf, dtype=float)
	wind_dir_xy = np.array([v_inf_array[0], v_inf_array[1], 0.0])
	norm = np.linalg.norm(wind_dir_xy)
	
	if norm < 1e-12:
		raise ValueError("v_inf has no horizontal component.")
	wind_dir_xy = wind_dir_xy / norm

	# Smallest enclosing circle radius from convex hull of footprint
	xy_coords = np.unique(vertices[:, :2], axis=0)
	hull_pts = xy_coords[ConvexHull(xy_coords).vertices]
	footprint_circumradius = 0.5 * cdist(hull_pts, hull_pts).max()

	cylinder_radius = r_factor * footprint_circumradius

	# Cylinder height and vertical extents
	ground_level = vertices[:, 2].min()
	mesh_top = vertices[:, 2].max()
	z_mid = 0.5 * (ground_level + mesh_top)
	
	cylinder_height = h_factor * (mesh_top - ground_level)

	# Horizontal center of bounding box
	xy_center = np.array([
		0.5 * (vertices[:, 0].min() + vertices[:, 0].max()),
		0.5 * (vertices[:, 1].min() + vertices[:, 1].max())
	])

	# Tilt: axis leans downstream with height
	tilt_gradient = np.tan(np.radians(tilt_deg))

	def get_ring_center(z_level):
		return np.append(xy_center + tilt_gradient * (z_level - z_mid) * wind_dir_xy[:2], z_level)

	# Grounded to earth (mesh bottom), clearance parameter removed
	z_bottom = ground_level
	z_top = z_bottom + cylinder_height

	# --- Intersection validation ---
	# The tilted cylinder axis goes from bottom_center to top_center.
	# For each building vertex we compute its distance from that axis line
	# and also check that the vertex falls within the axial extents of
	# the cylinder (i.e. between z_bottom and z_top).
	bottom_center = get_ring_center(z_bottom)
	top_center = get_ring_center(z_top)

	axis_vec = top_center - bottom_center          # vector along the cylinder axis
	axis_len_sq = np.dot(axis_vec, axis_vec)       # squared length of the axis

	# Project every vertex onto the axis to get a scalar t in [0, 1]
	diff = vertices - bottom_center                # (N, 3) offset from axis base
	t = (diff @ axis_vec) / axis_len_sq            # (N,) scalar projections

	# Closest point on the axis segment for each vertex
	t_clamped = np.clip(t, 0.0, 1.0)
	closest = bottom_center + t_clamped[:, np.newaxis] * axis_vec  # (N, 3)

	# Perpendicular distance from each vertex to the axis
	radial_dist = np.linalg.norm(vertices - closest, axis=1)       # (N,)

	# A vertex is outside the cylinder when its radial distance exceeds the
	# radius, within the axial extent of the cylinder.
	outside_mask = (radial_dist > cylinder_radius) & (t >= 0.0) & (t <= 1.0)

	if outside_mask.any():
		n_outside = outside_mask.sum()
		raise ValueError(
			f"Building geometry is not fully contained within the wake cylinder: "
			f"{n_outside} vertex/vertices found outside the cylinder volume. "
			f"Increase r_factor (currently {r_factor}) or h_factor (currently {h_factor}) "
			f"so the cylinder fully encloses the building."
		)

	return get_ring_center(z_bottom), get_ring_center(z_top), cylinder_radius

def visualize_points(pts: np.ndarray, point_size: int = 5, color: str = "cyan") -> None:
	"""
	Visualize a point cloud from an (N, 3) numpy array.
 
	Args:
		pts:        numpy array of shape (N, 3) with X, Y, Z columns.
		point_size: size of each rendered point.
		color:      color of the points (name or hex string).
	"""
	cloud = pv.PolyData(pts)
 
	plotter = pv.Plotter(window_size=[900, 700])
	plotter.set_background("black")
	plotter.add_points(cloud, color=color, point_size=point_size, render_points_as_spheres=True)

	# Add a ground plane at z=0 for context
	bounds = cloud.bounds
	cx = (bounds[0] + bounds[1]) / 2
	cy = (bounds[2] + bounds[3]) / 2
	size = max(bounds[1] - bounds[0], bounds[3] - bounds[2]) * 3
	ground = pv.Plane(center=(cx, cy, 0), direction=(0, 0, 1), i_size=size, j_size=size)
	plotter.add_mesh(ground, color="gray", opacity=0.2)

	plotter.add_axes()
	plotter.show()



if __name__ == "__main__":
	STL_PATH        = r"input_stls/Aerospecial_building4.stl"
	V_INF           = np.array([0.0, 13.6, 0.0])   
	R_FACTOR        = 3    # cylinder radius = R_FACTOR * building footprint circumradius
	H_FACTOR        = 1.4    # cylinder height = H_FACTOR * building height
	TILT_DEG        = 23.0   # downstream wake tilt in degrees
	import trimesh
	stl_mesh = trimesh.load_mesh(STL_PATH)
	bot,top,radius = calculate_wake_cylinder_parameters(
		stl_mesh, R_FACTOR, H_FACTOR, V_INF, tilt_deg=TILT_DEG
	)
	coords = generate_cylindrical_sampling_coordinates(
		bot, top, radius,
		z_clearance=.1,
		side_points=270,
		top_points=30,
		)
	visualize_points(coords, point_size=5)

	mesh = generate_momentum_integration_mesh(
    bot, top, radius,
    total_points=100_000,
    cap_top=True,
	)

	# Compute and visualize normals to verify orientation (consistent with momentum.py)
	mesh = mesh.compute_normals(
		cell_normals=True,
		point_normals=False,
		consistent_normals=True,
		auto_orient_normals=True,
	)

	plotter = pv.Plotter()
	plotter.add_mesh(mesh, show_edges=True, opacity=0.5, color="white")

	# Add ground plane
	size = radius * 15
	ground = pv.Plane(center=(bot[0], bot[1], 0), direction=(0, 0, 1), i_size=size, j_size=size)
	plotter.add_mesh(ground, color="gray", opacity=0.2)

	# Visualize cell normals using arrows at cell centers.
	# A factor of 5.0 is used to make them visible relative to the cylinder size.
	centers = mesh.cell_centers()
	centers["Normals"] = mesh.cell_data["Normals"]
	arrow_scale = radius * 0.15  # 15% of cylinder radius, adjust to taste
	arrows = centers.glyph(orient="Normals", scale=False, factor=arrow_scale)

	plotter.add_mesh(arrows, color="red")
	plotter.add_axes()
	plotter.show()
	
# 	  bottom center : [  6.52458984 110.09721077   0.        ]
#   top center    : [  6.52458984 141.35553624  73.64      ]
#   radius        : 76.2963 m
	

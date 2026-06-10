from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import numpy as np
import pyvista as pv


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
    total_points: int = 100_000,
    cap_bottom: bool = False,
    cap_top: bool = False,
    z_clearance: float = 0.1
) -> np.ndarray:
    """
    Build an oblique cylinder point cloud: horizontal flat caps, slanted wall.
    Returns an (N, 3) numpy array of coordinates.
    """
    center_bot = np.asarray(center_bot, dtype=float)
    center_top = np.asarray(center_top, dtype=float)

    cx_b, cy_b, z_bottom = center_bot
    cx_t, cy_t, z_top    = center_top

    # Apply z-clearance offset to the vertical bounds
    z_bottom += z_clearance
    z_top += z_clearance

    axis_vec    = center_top - center_bot
    axis_length = float(np.linalg.norm(axis_vec))

    n_caps       = int(cap_top) + int(cap_bottom)
    lateral_area = 2 * np.pi * radius * axis_length
    cap_area     = np.pi * radius ** 2
    total_area   = lateral_area + n_caps * cap_area

    s = np.sqrt(total_area / total_points)

    T         = max(4, round(2 * np.pi * radius / s))
    Z         = max(1, round(axis_length / s))
    cap_r_res = max(1, round(radius / s))

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

    # ------------------------------------------------------------------
    # 2. Cap interior vertices
    # ------------------------------------------------------------------
    all_pts_list = [lat_pts]

    if cap_bottom or cap_top:
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

    if cap_bottom:
        bot_cap_pts = make_cap_pts(cx_b, cy_b, z_bottom, shifts[0])
        all_pts_list.append(bot_cap_pts)

    if cap_top:
        top_cap_pts = make_cap_pts(cx_t, cy_t, z_top, shifts[-1])
        all_pts_list.append(top_cap_pts)

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

    return get_ring_center(z_bottom), get_ring_center(z_top), cylinder_radius
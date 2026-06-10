import os
import numpy as np
import numpy.random as rnd
import scipy as sp
import pandas as pd
import trimesh
from scipy.stats import qmc

from INTERP.interpolation import build_cfd_sampler


seed = 7
rnd.seed(seed)


# =============================================================================
# Geometry helpers
# =============================================================================

def _mesh_reject_mask(points, stl_mesh, epsilon=0.02, use_signed_distance=True):
    """
    Return boolean mask where True means point is valid.

    Rejects points inside the mesh and points closer than epsilon to its
    surface. Falls back to nearest-vertex distance if trimesh proximity
    isn't available.
    """
    points = np.asarray(points, dtype=float)
    valid = np.ones(points.shape[0], dtype=bool)

    try:
        valid &= ~stl_mesh.contains(points)
    except Exception:
        pass

    if use_signed_distance:
        try:
            query = trimesh.proximity.ProximityQuery(stl_mesh)
            signed_dist = query.signed_distance(points)
            valid &= np.abs(signed_dist) > epsilon
            return valid
        except Exception:
            pass

    tree = sp.spatial.cKDTree(np.asarray(stl_mesh.vertices))
    distances, _ = tree.query(points)
    valid &= distances > epsilon
    return valid


def _oblique_cylinder_points(stl_mesh,
                             r_factor,
                             h_factor,
                             v_inf,
                             tilt_deg=23.0,
                             clearance=0.1,
                             n_points=300,
                             front_frac=0.5,
                             front_half_angle_deg=45.0,
                             top_cap=False,
                             top_cap_frac=0.25,
                             seed=7):

    V = np.asarray(stl_mesh.vertices)

    # downstream (streamwise) and cross-wind unit vectors in the horizontal plane
    v = np.asarray(v_inf, float)
    s = np.array([v[0], v[1], 0.0])
    n = np.linalg.norm(s)
    if n < 1e-12:
        raise ValueError("v_inf has no horizontal component; cannot define wake direction.")
    s = s / n                              # downstream (+wake) direction
    c = np.array([-s[1], s[0], 0.0])       # cross-wind direction (horizontal)

    width_body = (V @ c).max() - (V @ c).min()   # cross-wind width
    height_body = V[:, 2].max() - V[:, 2].min()  # vertical height
    char_size = max(width_body, height_body)

    R = r_factor * char_size
    H = h_factor * char_size

    z_lo = V[:, 2].min()
    z_hi = V[:, 2].max()
    z_mid = 0.5 * (z_lo + z_hi)

    P0 = np.array([0.5 * (V[:, 0].min() + V[:, 0].max()),
                   0.5 * (V[:, 1].min() + V[:, 1].max())])   # horizontal center

    shift_per_height = np.tan(np.radians(tilt_deg))
    z_bottom = z_lo + clearance
    z_top = z_bottom + H

    def ring_center_xy(z):
        return P0 + shift_per_height * (z - z_mid) * s[:2]   # leans downstream with height

    bottom_center = np.array([*ring_center_xy(z_bottom), z_bottom])
    top_center = np.array([*ring_center_xy(z_top), z_top])

    half = np.radians(front_half_angle_deg)

    n_cap = int(round(top_cap_frac * n_points)) if top_cap else 0
    n_side = n_points - n_cap
    n_front = int(round(front_frac * n_side))
    n_back = n_side - n_front

    def shell_lhs(nn, phi_min, phi_max, sd):
        if nn <= 0:
            return np.empty((0, 3))
        u = qmc.LatinHypercube(d=2, seed=sd).random(nn)
        phi = phi_min + u[:, 0] * (phi_max - phi_min)
        z = z_bottom + u[:, 1] * H
        rc = P0[None, :] + shift_per_height * (z - z_mid)[:, None] * s[None, :2]
        # phi=0 -> +s (downstream wake); sweep into cross-wind via c
        offset = R * (np.cos(phi)[:, None] * s[None, :2] + np.sin(phi)[:, None] * c[None, :2])
        xy = rc + offset
        return np.column_stack([xy[:, 0], xy[:, 1], z])

    def cap_lhs(nn, sd):
        if nn <= 0:
            return np.empty((0, 3))
        u = qmc.LatinHypercube(d=2, seed=sd).random(nn)
        rho = R * np.sqrt(u[:, 0])
        theta = 2.0 * np.pi * u[:, 1]
        cxy = ring_center_xy(z_top)            # tilted disk center at cap height
        xy = cxy[None, :] + (np.cos(theta)[:, None] * s[None, :2]
                             + np.sin(theta)[:, None] * c[None, :2]) * rho[:, None]
        return np.column_stack([xy[:, 0], xy[:, 1], np.full(nn, z_top)])

    # phi ~ 0 is the wake side; this wedge gets the dense sampling.
    front = shell_lhs(n_front, -half, half, seed)
    back = shell_lhs(n_back, half, 2.0 * np.pi - half, seed + 1)
    cap = cap_lhs(n_cap, seed + 2)
    points = np.vstack([front, back, cap])

    return points, R, bottom_center, top_center


def _drone_array_points(stl_mesh,
                        bounds,
                        tilt_deg=20.0,       #theta, tilt of plane
                        standoff=None,        #d0, if we want to fix flow distance building <-> array
                        standoff_factor=1.0,  #if we want to use a factor*length of building for flow distance building <-> array
                        z_floor = None,          #bottom-row altitude, if none the first row will have a height of domain floor + clearance
                        z_clearance=0.0,
                        n_rows = 9,
                        n_cols = 9,
                        width_factor = 2.5, #target width of array = width_factor * building width 
                        height_factor = 2.0, #vertical coverage target = height_factor * building height
                        domain_margin_frac=0.02, #fractional margin to stay within CFD domain

                        return_pool = False,
                        prior_fn = None,
                        v_inf = None,
                        freestream_frac = 0.05,
                        pool_pad_factor = 1.5,
                        pool_n_per_axis=10,
                        keepout = 5,
                        ):
    #calculate the target of sampling points for a check at the end and check that we at least have one row and column
    if n_rows < 1 or n_cols < 1:
        raise ValueError("n_rows and n_cols must be >=1")
    
    #calculate cosine (ct) and sin (st) of theta once
    theta   = np.radians(tilt_deg)
    ct      = np.cos(theta)
    st      = np.sin(theta)

    #define building references
    V  = np.asarray(stl_mesh.vertices) #open the vertices of the building
    vertices_x, vertices_y, vertices_z = V[:,0], V[:,1], V[:,2]
    back_station = vertices_x.max() #furthest downstream building point
    lat_center = 0.5 * (vertices_y.min() + vertices_y.max()) #centers array around building
    width_body = vertices_y.max() - vertices_y.min()
    height_body = vertices_z.max() - vertices_z.min()
    char_size = max(width_body, height_body) #characteristic size for d0

    #bounds is a (3,2) array sample() build. row 0 =x, ... and first column min, second column max
    x0, x1 = bounds[0, 0], bounds[0, 1]   # data x range (downstream)
    y0, y1 = bounds[1, 0], bounds[1, 1]   # data y range (sideways)
    z0, z1 = bounds[2, 0], bounds[2, 1]   # data z range (height)
    
    #calculate margins to not get a sample on the boundary of the cdf
    mx = domain_margin_frac * (x1 - x0)
    my = domain_margin_frac * (y1 - y0)
    mz = domain_margin_frac * (z1 - z0)

    x_legal_max         = x1 - mx
    z_legal_max         = z1 - mz
    y_legal_min         = y0 + my
    y_legal_max         = y1 - my
    z_legal_min         = z0 + mz

    #define the anchor: the single bottom row, middle-column drone
    if standoff is not None:
        d0 = standoff
    else: 
        d0 = standoff_factor * char_size
    x_anchor = back_station + d0
    y_anchor = lat_center
    
    if z_floor is None:
        z_anchor = z_legal_min + z_clearance
    else: 
        z_anchor = z_floor
    
    #make sure the width and length fit into the cfd domain
    W_target = width_factor * width_body
    W_max = 2.0 * min(y_anchor - y_legal_min, y_legal_max - y_anchor) 
    W = min(W_target, W_max)

    L_target = (height_factor * height_body / ct) if ct > 1e-9 else np.inf
    L_vert_constraint = (z_legal_max - z_anchor) / ct if ct > 1e-9 else np.inf
    L_hor_constraint = ( x_legal_max - x_anchor) / st if st > 1e-9 else np.inf
    L = min(L_target, L_vert_constraint, L_hor_constraint)

    if W <= 0.0 or L <= 0.0:
        raise RuntimeError(
            f"Degenerate array footprint: W={W:.4g}, L={L:.4g}. "
            "The CFD domain is too tight for this anchor/tilt."
        )
    
    #make grid
    w_coords = np.linspace(-0.5*W, 0.5*W, n_cols) #sideways positions, centered around anchor
    l_coords = np.linspace(0.0, L, n_rows)        #positions up the slant, 0 is the anchor drone
    
    ww, ll = np.meshgrid(w_coords, l_coords)
    ww_flat = ww.ravel()
    ll_flat = ll.ravel()

    x_drones = x_anchor + ll_flat * st
    y_drones = y_anchor + ww_flat
    z_drones = z_anchor + ll_flat * ct
    pts = np.column_stack([x_drones, y_drones, z_drones])

    #reporting: did the cfd domain force the array smaller than we wanted?
    width_clamped = W < W_target - 1e-9
    slant_clamped = L < L_target - 1e-9



    if not return_pool:
        return pts
    if prior_fn is None or v_inf is None:
        raise ValueError("return_pool=True needs prior_fn and v_inf for the freestream mask.")
    
    #how far around the building the building reaches
    pad = pool_pad_factor * char_size
    
    #outer box, cfd domain minus margin
    out_y0, out_y1 = y0 + my, y1 - my
    out_z0, out_z1 = z0 + mz, z1 - mz
    out_x0, out_x1 = x0 + mx, x1 - mx                   #uncomment if we also want drones in the wake
    # out_x0, out_x1 = x0 + mx, vertices_x.max()        #uncomment if we dont want drones in the wake

    #inner box, dibt want too close to the building
    in_x0 = vertices_x.min() - keepout
    in_x1 = vertices_x.max() + keepout
    in_y0 = vertices_y.min() - keepout
    in_y1 = vertices_y.max() + keepout
    in_z0 = vertices_z.min() - keepout
    in_z1 = vertices_z.max() + keepout
        
    # fill the box with a regular lattice of raw candidates
    px = np.linspace(out_x0, out_x1, pool_n_per_axis)
    py = np.linspace(out_y0, out_y1, pool_n_per_axis)
    pz = np.linspace(out_z0, out_z1, pool_n_per_axis)
    pxx, pyy, pzz = np.meshgrid(px, py, pz, indexing="ij")
    pool = np.column_stack([pxx.ravel(), pyy.ravel(), pzz.ravel()])


    # exclude the points in inner box
    inside_building_box = (
        (pool[:, 0] > in_x0) & (pool[:, 0] < in_x1) &
        (pool[:, 1] > in_y0) & (pool[:, 1] < in_y1) &
        (pool[:, 2] > in_z0) & (pool[:, 2] < in_z1)
    )
    pool = pool[~inside_building_box]

    #get the points out where potential flow is close to the freestream flow
    tol = freestream_frac * np.linalg.norm(np.asarray(v_inf))
    pool = pool[_freestream_mask(pool, prior_fn, v_inf, tol)]

    return pts, pool

def _freestream_mask(points, prior_fn, v_inf, tol):
    points = np.asarray(points)

    # Potential flow at each candidate point
    prior = np.asarray(prior_fn(points))

    # Use the freestream as a full vector so any wind direction works.
    V_inf = np.asarray(v_inf)

    # sqrt((du)^2 + (dv)^2 + (dw)^2) for each row -> deviation from freestream (m/s).
    deviation = np.linalg.norm(prior - V_inf, axis=1)

    # True = disturbed, False = essentially freestream (drop).
    return deviation > tol

def sample(
    field_path,
    stl_mesh,
    samples=None,
    sample_method="CSV",
    sample_config=None,
    v_inf=None,
    epsilon=0.02,
    num_samples=150,
    use_signed_distance=True,
    oversample_factor=4,
    max_random_iters=100,
):
    method = sample_method
    config = sample_config
    cylinder_geom = None  

    # ---- load CFD field: support both .csv and .pkl/.pickle ----
    ext = os.path.splitext(field_path)[1].lower()
    if ext in (".pkl", ".pickle"):
        print('Reading pickle...')
        cfd = pd.read_pickle(field_path)
    elif ext == ".csv":
        print('Reading csv...')
        cfd = pd.read_csv(field_path)
    else:
        raise ValueError(f"Unsupported CFD file type {ext!r} (expected .csv or .pkl)")

    cfd.columns = cfd.columns.str.strip()

    required = [
        "x-coordinate", "y-coordinate", "z-coordinate",
        "x-velocity", "y-velocity", "z-velocity", "pressure",
    ]
    missing = [c for c in required if c not in cfd.columns]
    if missing:
        raise ValueError(f"Missing required columns in CFD CSV: {missing}")

    print('Extracting coords & vals...')
    source_coords = cfd[["x-coordinate", "y-coordinate", "z-coordinate"]].to_numpy(float)
    source_values = cfd[["x-velocity", "y-velocity", "z-velocity", "pressure"]].to_numpy(float)

    bounds = np.array([
        [source_coords[:, 0].min(), source_coords[:, 0].max()],
        [source_coords[:, 1].min(), source_coords[:, 1].max()],
        [source_coords[:, 2].min(), source_coords[:, 2].max()],
    ])

    if ext == ".csv":
        print("Building interpolator (griddata)...")
        def sample_dat_shi(points):
            return sp.interpolate.griddata(
                source_coords, source_values, points,
                method="linear", fill_value=np.nan,
            )
        del cfd
    else:  # .pkl / .pickle large case
        print("Building interpolator (fast KD-tree sampler)...")
        sample_dat_shi = build_cfd_sampler(cfd, n_points=8, sharpness=2)
        del cfd

    def reject(points):
        return points[_mesh_reject_mask(
            points, stl_mesh, epsilon=epsilon, use_signed_distance=use_signed_distance,
        )]

    print('Collecting points of interest...')
    if method == "drone_array":
        points = reject(_drone_array_points(stl_mesh, bounds, **(config or {})))
        if len(points) == 0:
            raise RuntimeError("drone_array produced no valid points.")

    elif method == "cylinder":
        if v_inf is None:
            raise ValueError("v_inf is required for cylinder sampling.")
        cyl_pts, R, bottom_center, top_center = _oblique_cylinder_points(
            stl_mesh, v_inf=v_inf, **(config or {}))
        cylinder_geom = {"R": R, "bottom_center": bottom_center, "top_center": top_center}
        points = reject(cyl_pts)
        if len(points) == 0:
            raise RuntimeError("cylinder produced no valid points.")

    elif method == "random":
        collected = []
        n = 0
        for _ in range(max_random_iters):
            if n >= num_samples:
                break
            batch = np.column_stack([
                rnd.uniform(bounds[0, 0], bounds[0, 1], num_samples * oversample_factor),
                rnd.uniform(bounds[1, 0], bounds[1, 1], num_samples * oversample_factor),
                rnd.uniform(bounds[2, 0], bounds[2, 1], num_samples * oversample_factor),
            ])
            safe = reject(batch)
            if len(safe):
                collected.append(safe)
                n += len(safe)
        if not collected:
            raise RuntimeError("Random sampling failed: no valid points.")
        points = np.vstack(collected)[:num_samples]

    elif method in ("CSV", "array"):
        if samples is None:
            raise ValueError(f"samples is required when method='{method}'.")
        if method == "CSV":
            tdf = pd.read_csv(samples)
            tdf.columns = tdf.columns.str.strip()
            cols = ["x-coordinate", "y-coordinate", "z-coordinate"]
            points = (tdf[cols] if cols[0] in tdf.columns else tdf.iloc[:, :3]).to_numpy(float)
        else:
            points = np.asarray(samples, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {points.shape}")

        in_bounds = np.all((points >= bounds[:, 0]) & (points <= bounds[:, 1]), axis=1)
        points = reject(points[in_bounds])
        if len(points) == 0:
            raise ValueError("No points left to sample.")

    else:
        raise ValueError(f"unknown method {method!r}")

    values = sample_dat_shi(points)

    columns = [
        "x-target", "y-target", "z-target",
        "x-velocity", "y-velocity", "z-velocity", "pressure",
    ]
    df = pd.DataFrame(np.hstack([points, values]), columns=columns)

    before = len(df)
    df = df.dropna().reset_index(drop=True)

    print(f"CFD source points:    {len(source_coords)}")
    print(f"Candidate points:     {before}")
    print(f"Dropped (NaN):        {before - len(df)}")
    print(f"Returned samples:     {len(df)}")

    if len(df) == 0:
        raise RuntimeError("All sampled points became NaN after interpolation.")

    return df, bounds, sample_dat_shi, cylinder_geom
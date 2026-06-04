import numpy as np
import numpy.random as rnd
import scipy as sp
import pandas as pd
import trimesh


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


# =============================================================================
# Drone-array sampling  
# =============================================================================
def _oblique_cylinder_points(stl_mesh,
                             r_factor,
                             h_factor,
                             tilt_deg=23.0,
                             clearance=5.0,
                             n_rings=10,
                             n_per_ring=72):
    V = np.asarray(stl_mesh.vertices)               #get corners of building
    width_body  = V[:, 1].max() - V[:, 1].min()     #ymax-ymin
    height_body = V[:, 2].max() - V[:, 2].min()     #zmax-zmin
    char_size   = max(width_body, height_body)      #get largest

    R = r_factor * char_size                        #get radius
    H = h_factor * char_size                        #get height

    z_lo, z_hi = V[:, 2].min(), V[:, 2].max()       #lowest point, highest point
    z_mid = 0.5 * (z_lo + z_hi)                     #mid height of building
    cx = 0.5 * (V[:, 0].min() + V[:, 0].max())      #x location of centroid
    cy = 0.5 * (V[:, 1].min() + V[:, 1].max())      #y location of centroid

    shift_per_height = np.tan(np.radians(tilt_deg)) #slope of oblique cylinder

    z_bottom = z_lo + clearance                     #bottom z-coordinate of cricle
    dz_bottom = z_bottom - z_mid                    #change between bottom and mid
    bottom_cx = cx + shift_per_height * dz_bottom   #change in x between bottom and mid bcs of tilt
    bottom_cy = cy                                  #y doesnt change

    z_rings = np.linspace(z_bottom, z_bottom + H, n_rings)          #get a spacing of rings
    phi = np.linspace(0.0, 2.0 * np.pi, n_per_ring, endpoint=False) #get a lot of different angles for the circle

    pts = []
    for z in z_rings:
        dz = z - z_bottom                                           #for every vertical height, calculate difference with bottom
        ring_cx = bottom_cx + shift_per_height * dz                 #calculate shift in x xcoordinate of centroid bcs of tilt
        ring_cy = bottom_cy                                         #calculate ycoordinate of centroid
        x = ring_cx + R * np.cos(phi)                               #make circle
        y = ring_cy + R * np.sin(phi)                               #make circle
        zc = np.full(n_per_ring, z)                                 #make array of n_per_ring times the z coordinate
        pts.append(np.column_stack([x, y, zc]))                     #add the points to the list

    return np.vstack(pts)

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

# =============================================================================
# Main sample()
# =============================================================================

def sample(
    field_path,
    stl_mesh,
    samples=None,
    method="CSV",
    epsilon=0.02,
    num_samples=400,
    use_signed_distance=True,
    oversample_factor=4,
    max_random_iters=100,
    drone_array_config=None,
):
    """
    Sample CFD velocity + pressure at target points.

    method:
        "CSV"          -- `samples` is a path to a CSV with x/y/z-coordinate
                          columns (or the first three columns).
        "array"        -- `samples` is an (N, 3) array of target coordinates.
        "random"       -- scatter num_samples points through the domain,
                          rejecting any that land inside / near the mesh.
        "drone_array"  -- generate a tilted drone grid via _drone_array_points;
                          pass generator kwargs through `drone_array_config`.

    Returns (results_df, bounds).
    """
    cfd_df = pd.read_csv(field_path)
    cfd_df.columns = cfd_df.columns.str.strip()

    required_cols = [
        "x-coordinate", "y-coordinate", "z-coordinate",
        "x-velocity", "y-velocity", "z-velocity", "pressure",
    ]
    missing = [c for c in required_cols if c not in cfd_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CFD CSV: {missing}")

    source_coords = cfd_df[
        ["x-coordinate", "y-coordinate", "z-coordinate"]
    ].values.astype(float)
    source_values = cfd_df[
        ["x-velocity", "y-velocity", "z-velocity", "pressure"]
    ].values.astype(float)

    x_min, x_max = source_coords[:, 0].min(), source_coords[:, 0].max()
    y_min, y_max = source_coords[:, 1].min(), source_coords[:, 1].max()
    z_min, z_max = source_coords[:, 2].min(), source_coords[:, 2].max()

    bounds = np.array(
        [[x_min, x_max], [y_min, y_max], [z_min, z_max]], dtype=float
    )

    # -------------------------------------------------------------------------
    # Build target points
    # -------------------------------------------------------------------------
    if method == "drone_array":
        cfg = drone_array_config or {}
        valid_points = _drone_array_points(stl_mesh, bounds, **cfg)
        # Reject anything that landed inside / too close to the body.
        valid_points = valid_points[
            _mesh_reject_mask(valid_points, stl_mesh, epsilon=epsilon,
                              use_signed_distance=use_signed_distance)
        ]
        if len(valid_points) == 0:
            raise RuntimeError("drone_array produced no valid points.")

    elif method == "random":
        rng_points = []
        collected = 0
        for _ in range(max_random_iters):
            if collected >= num_samples:
                break
            remaining = num_samples - collected
            batch_size = max(remaining * oversample_factor, remaining)
            batch = np.column_stack((
                rnd.uniform(x_min, x_max, batch_size),
                rnd.uniform(y_min, y_max, batch_size),
                rnd.uniform(z_min, z_max, batch_size),
            ))
            safe = batch[
                _mesh_reject_mask(batch, stl_mesh, epsilon=epsilon,
                                  use_signed_distance=use_signed_distance)
            ]
            if len(safe):
                rng_points.append(safe)
                collected += len(safe)
        if collected == 0:
            raise RuntimeError("Random sampling failed: no valid points.")
        valid_points = np.vstack(rng_points)[:num_samples]

    elif method in ("CSV", "array"):
        if method == "CSV":
            if samples is None:
                raise ValueError("samples must be a CSV path when method='CSV'.")
            target_df = pd.read_csv(samples)
            target_df.columns = target_df.columns.str.strip()
            if "x-coordinate" in target_df.columns:
                target_points = target_df[
                    ["x-coordinate", "y-coordinate", "z-coordinate"]
                ].values.astype(float)
            else:
                target_points = target_df.iloc[:, :3].values.astype(float)
        else:  # array
            if samples is None:
                raise ValueError("samples must be an array when method='array'.")
            target_points = np.asarray(samples, dtype=float)

        if target_points.ndim == 1:
            target_points = target_points.reshape(1, -1)
        if target_points.ndim != 2 or target_points.shape[1] != 3:
            raise ValueError(
                f"target_points must have shape (N, 3), got {target_points.shape}"
            )

        bounds_mask = (
            (target_points[:, 0] >= x_min) & (target_points[:, 0] <= x_max)
            & (target_points[:, 1] >= y_min) & (target_points[:, 1] <= y_max)
            & (target_points[:, 2] >= z_min) & (target_points[:, 2] <= z_max)
        )
        valid_points = target_points[bounds_mask]
        if len(valid_points):
            valid_points = valid_points[
                _mesh_reject_mask(valid_points, stl_mesh, epsilon=epsilon,
                                  use_signed_distance=use_signed_distance)
            ]
        if len(valid_points) == 0:
            raise ValueError("No points left to sample.")

    else:
        raise ValueError(
            "method must be 'CSV', 'array', 'random', or 'drone_array'."
        )

    # -------------------------------------------------------------------------
    # Interpolate CFD values at target points
    # -------------------------------------------------------------------------
    interpolated_values = sp.interpolate.griddata(
        points=source_coords,
        values=source_values,
        xi=valid_points,
        method="linear",
        fill_value=np.nan,
    )

    columns = [
        "x-target", "y-target", "z-target",
        "x-velocity", "y-velocity", "z-velocity", "pressure",
    ]
    results_df = pd.DataFrame(
        np.hstack((valid_points, interpolated_values)), columns=columns
    )

    before = len(results_df)
    results_df = results_df.dropna().reset_index(drop=True)
    dropped = before - len(results_df)

    print("\nSampling diagnostics")
    print("--------------------")
    print(f"CFD source points:          {len(source_coords)}")
    print(f"Candidate valid points:     {before}")
    print(f"Dropped after griddata NaN: {dropped}")
    print(f"Returned samples:           {len(results_df)}")
    print(f"Bounds:\n{bounds}")

    if len(results_df) == 0:
        raise RuntimeError("All sampled points became NaN after interpolation.")

    return results_df, bounds
import numpy as np
import numpy.random as rnd
import scipy as sp
import pandas as pd
import trimesh

import os
# Force CPU before importing JAX.
os.environ["JAX_PLATFORMS"] = "gpu"

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

seed = 7
rnd.seed(seed)


# =============================================================================
# Geometry helpers
# =============================================================================

def _mesh_reject_mask(points, stl_mesh, epsilon=0.02, use_signed_distance=True):
    """
    Return boolean mask where True means point is valid.

    Rejects:
        - points inside mesh
        - points closer than epsilon to the mesh surface

    Falls back to nearest-vertex distance if trimesh proximity is unavailable.
    """
    points = np.asarray(points, dtype=float)
    valid = np.ones(points.shape[0], dtype=bool)

    try:
        inside = stl_mesh.contains(points)
        valid &= ~inside
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

    wall_points = np.asarray(stl_mesh.vertices)
    wall_tree = sp.spatial.cKDTree(wall_points)
    distances, _ = wall_tree.query(points)
    valid &= distances > epsilon
    return valid


def _surface_distance(points, stl_mesh):
    """Unsigned distance from each point to the mesh surface (N,)."""
    points = np.asarray(points, dtype=float)
    try:
        query = trimesh.proximity.ProximityQuery(stl_mesh)
        return np.abs(query.signed_distance(points))
    except Exception:
        tree = sp.spatial.cKDTree(np.asarray(stl_mesh.vertices))
        d, _ = tree.query(points)
        return d


def _body_metrics(stl_mesh):
    """
    Return geometric descriptors used to shape the wake sampling.

    Returns dict with:
        bmin, bmax: bounding box corners (3,)
        length_x:   streamwise extent
        width_y:    cross-stream extent
        height_z:   vertical extent
        x_base:     downstream-most x (flat face of the prism)
        x_apex:     upstream-most x (leading edge)
        diameter:   characteristic cross-stream size (max of y/z extent)
    """
    b = np.asarray(stl_mesh.bounds, dtype=float)
    bmin, bmax = b[0], b[1]
    ext = bmax - bmin
    return {
        "bmin": bmin,
        "bmax": bmax,
        "length_x": float(ext[0]),
        "width_y": float(ext[1]),
        "height_z": float(ext[2]),
        "x_base": float(bmax[0]),
        "x_apex": float(bmin[0]),
        "diameter": float(max(ext[1], ext[2])),
    }


# =============================================================================
# Cylinder-wall sampling (CV method)
# =============================================================================

def _cylinder_wall_points(
    stl_mesh,
    num_samples,
    epsilon,
    use_signed_distance,
    rng,
    cylinder_radius=None,
    z_min=0.0,
    z_max=10,
    oversample=6,
):
    """
    Sample points on the lateral wall of a vertical cylinder centred at
    (x=0, y=0), spanning z in [z_min, z_max].

    The cylinder radius defaults to the larger of the body's bounding-box
    half-extents in x/y plus a small clearance (1.5x the body half-diagonal
    in the xy plane), so it always wraps around the geometry.  A custom
    radius can be passed via `cylinder_radius`.

    Points are distributed as follows to give good angular & vertical
    coverage while concentrating effort near the body:

        • Uniform-random on the cylindrical surface
          (phi in [0, 2π), z in [z_min, z_max]).
        • Stratified z-bands so no tall strip is left empty.
        • After placing, any point that falls inside or within `epsilon`
          of the mesh is dropped (safety net only – the cylinder should
          already sit outside the body).

    Parameters
    ----------
    stl_mesh        : trimesh.Trimesh
    num_samples     : target number of returned points
    epsilon         : min clearance from mesh surface
    use_signed_distance : passed to _mesh_reject_mask
    rng             : numpy Generator
    cylinder_radius : float or None  (auto-computed when None)
    z_min, z_max    : float  axial extent of the cylinder
    oversample      : int    generate this many × num_samples candidates
                             before rejection, to absorb any mesh conflicts

    Returns
    -------
    pts : (M, 3) array,  M <= num_samples
    """
    bm = _body_metrics(stl_mesh)

    # Auto-radius: 1.5 × half-diagonal of the body's xy bounding box.
    if cylinder_radius is None:
        half_x = 0.5 * bm["length_x"]
        half_y = 0.5 * bm["width_y"]
        cylinder_radius = 1.5 * np.hypot(half_x, half_y)
        cylinder_radius = max(cylinder_radius, bm["diameter"])  # at least D

    r = float(cylinder_radius)
    n_cand = num_samples * oversample

    # Stratified z so we always cover the full height.
    n_bands = max(10, num_samples // 5)
    z_edges = np.linspace(z_min, z_max, n_bands + 1)
    per_band = n_cand // n_bands
    remainder = n_cand - per_band * n_bands

    z_cand_parts = []
    for i in range(n_bands):
        n_here = per_band + (1 if i < remainder else 0)
        z_cand_parts.append(rng.uniform(z_edges[i], z_edges[i + 1], n_here))
    z_cand = np.concatenate(z_cand_parts)

    # Uniform angle.
    phi = rng.uniform(0.0, 2.0 * np.pi, len(z_cand))

    x_cand = r * np.cos(phi)
    y_cand = r * np.sin(phi)

    pts = np.column_stack([x_cand, y_cand, z_cand])

    # Safety rejection against the mesh (should rarely trigger).
    keep = _mesh_reject_mask(
        pts, stl_mesh, epsilon=epsilon, use_signed_distance=use_signed_distance
    )
    pts = pts[keep]

    # Shuffle so that trimming to num_samples gives uniform coverage.
    idx = rng.permutation(len(pts))
    pts = pts[idx]

    print("\nCylinder-wall sampling (CV method)")
    print("------------------------------------")
    print(f"Cylinder radius:  {r:.4g}")
    print(f"Axial range:      z = {z_min:.3g} … {z_max:.3g}")
    print(f"Candidates:       {n_cand}  (oversample ×{oversample})")
    print(f"After rejection:  {len(pts)}")
    print(f"Returning:        {min(len(pts), num_samples)}")

    return pts[:num_samples]


def sample(
    field_path,
    stl_mesh,
    samples=None,
    method="CSV",
    epsilon=0.02,
    num_samples=150,
    use_signed_distance=True,
    oversample_factor=4,
    max_random_iters=100,
    max_points=150,
    prior_fn=None,
    cylinder_radius=None,
    cylinder_z_min=0.0,
    cylinder_z_max=7.5,
):

    cfd_df = pd.read_csv(field_path)
    cfd_df.columns = cfd_df.columns.str.strip()
    print(f'CFD Loaded. Array shape:{cfd_df.shape}')
    required_cols = [
        "x-coordinate", "y-coordinate", "z-coordinate",
        "x-velocity", "y-velocity", "z-velocity", "pressure",
    ]
    missing = [c for c in required_cols if c not in cfd_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CFD CSV: {missing}")

    print('Converting load to jax numpy arrays...')
    # source_coords = jnp.asarray(cfd_df[["x-coordinate", "y-coordinate", "z-coordinate"]].values.astype(float))
    # source_values = jnp.asarray(cfd_df[["x-velocity", "y-velocity", "z-velocity", "pressure"]].values.astype(float))

    x_axis = jnp.array(np.sort(cfd_df["x-coordinate"].unique()).astype(np.float32))
    y_axis = jnp.array(np.sort(cfd_df["y-coordinate"].unique()).astype(np.float32))
    z_axis = jnp.array(np.sort(cfd_df["z-coordinate"].unique()).astype(np.float32))

    x_min, x_max = x_axis[:, 0].min(), x_axis[:, 0].max()
    y_min, y_max = y_axis[:, 1].min(), y_axis[:, 1].max()
    z_min, z_max = z_axis[:, 2].min(), z_axis[:, 2].max()

    bounds = np.array(
        [[x_min, x_max], [y_min, y_max], [z_min, z_max]], dtype=float
    )

    cv_cap = int(min(num_samples, max_points))
    rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------------
    # Target points
    # -------------------------------------------------------------------------
    print(f'Collecting points of interest (method: {method})')
    if method == "cv":
        valid_points = _cylinder_wall_points(
            stl_mesh=stl_mesh,
            num_samples=cv_cap,
            epsilon=epsilon,
            use_signed_distance=use_signed_distance,
            rng=rng,
            cylinder_radius=cylinder_radius,
            z_min=cylinder_z_min,
            z_max=cylinder_z_max,
        )

        if len(valid_points) == 0:
            raise RuntimeError(
                "CV cylinder sampling produced no valid points. "
                "Check mesh/CFD units, cylinder radius, and epsilon."
            )

    elif method == "random":
        valid_points_list = []
        collected = 0
        for _ in range(max_random_iters):
            if collected >= num_samples:
                break
            remaining = num_samples - collected
            batch_size = max(remaining * oversample_factor, remaining)
            xr = rnd.uniform(x_min, x_max, batch_size)
            yr = rnd.uniform(y_min, y_max, batch_size)
            zr = rnd.uniform(z_min, z_max, batch_size)
            batch = np.column_stack((xr, yr, zr))
            mask = _mesh_reject_mask(
                batch, stl_mesh, epsilon=epsilon,
                use_signed_distance=use_signed_distance,
            )
            safe = batch[mask]
            if len(safe) > 0:
                valid_points_list.append(safe)
                collected += len(safe)
        if collected == 0:
            raise RuntimeError("Random sampling failed: no valid points.")
        valid_points = np.vstack(valid_points_list)[:num_samples]

    else:
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
        elif method == "array":
            if samples is None:
                raise ValueError("samples must be an array when method='array'.")
            target_points = np.asarray(samples, dtype=float)
        else:
            raise ValueError("method must be 'CSV', 'array', 'random', or 'cv'.")

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
        if len(valid_points) > 0:
            mask = _mesh_reject_mask(
                valid_points, stl_mesh, epsilon=epsilon,
                use_signed_distance=use_signed_distance,
            )
            valid_points = valid_points[mask]
        if len(valid_points) == 0:
            raise ValueError("No points left to sample.")

    # -------------------------------------------------------------------------
    # Interpolate CFD values at target points
    # -------------------------------------------------------------------------
    print('Collecting measurements at points of interest (jax interpolation)...')
    # interpolated_values = sp.interpolate.griddata(
    #     points=source_coords,
    #     values=source_values,
    #     xi=valid_points,
    #     method="linear",
    #     fill_value=np.nan,
    # )

    print("Inferring grid dimensions...")
    nx = int(cfd_df["x-coordinate"].nunique())
    ny = int(cfd_df["y-coordinate"].nunique())
    nz = int(cfd_df["z-coordinate"].nunique())

    # Reshape flat columns into (nx, ny, nz) grids on GPU
    u_grid = jnp.array(cfd_df["x-velocity"].values.astype(np.float32).reshape(nx, ny, nz))
    v_grid = jnp.array(cfd_df["y-velocity"].values.astype(np.float32).reshape(nx, ny, nz))
    w_grid = jnp.array(cfd_df["z-velocity"].values.astype(np.float32).reshape(nx, ny, nz))
    p_grid = jnp.array(cfd_df["pressure"].values.astype(np.float32).reshape(nx, ny, nz))
    del cfd_df
    
    MAX_N = max_points
    
    @jax.jit
    def interpolate(xq, yq, zq):
        # Convert physical coords → fractional grid indices
        def to_idx(q, axis):
            return (q - axis[0]) / (axis[-1] - axis[0]) * (len(axis) - 1)

        coords = jnp.stack([
            to_idx(xq, x_axis),
            to_idx(yq, y_axis),
            to_idx(zq, z_axis),
        ])  # shape (3, MAX_N)

        return jnp.stack([
            map_coordinates(u_grid, coords, order=1, mode='nearest'),
            map_coordinates(v_grid, coords, order=1, mode='nearest'),
            map_coordinates(w_grid, coords, order=1, mode='nearest'),
            map_coordinates(p_grid, coords, order=1, mode='nearest'),
        ], axis=1)  # (MAX_N, 4)

    # -------------------------------------------------------------------------
    # Interpolate CFD values at valid_points — padded to fixed size
    # -------------------------------------------------------------------------
    print('Interpolating CFD values at query points (GPU)...')
    N = len(valid_points)
    xq = np.pad(valid_points[:, 0].astype(np.float32), (0, MAX_N - N))
    yq = np.pad(valid_points[:, 1].astype(np.float32), (0, MAX_N - N))
    zq = np.pad(valid_points[:, 2].astype(np.float32), (0, MAX_N - N))

    result = np.asarray(interpolate(jnp.array(xq), jnp.array(yq), jnp.array(zq)))
    interpolated_values = result[:N]
    
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

    # Enforce hard cap after NaN drop.
    if len(results_df) > max_points:
        results_df = results_df.iloc[:max_points].reset_index(drop=True)

    print("\nSampling diagnostics")
    print("--------------------")
    # print(f"CFD source points:          {len(source_coords)}")
    print(f"Candidate valid points:     {before}")
    print(f"Dropped after griddata NaN: {dropped}")
    print(f"Returned samples:           {len(results_df)} (cap {max_points})")
    print(f"Bounds:\n{bounds}")

    if len(results_df) == 0:
        raise RuntimeError(
            "All sampled points became NaN after interpolation."
        )

    return results_df, bounds
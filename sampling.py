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

    Rejects:
        - points inside mesh
        - points closer than epsilon to the mesh surface

    Requires trimesh proximity/contains dependencies such as rtree for robust
    mesh queries. Falls back to nearest-vertex distance only for clearance.
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
        dist, _ = tree.query(points)
        return dist


def _body_metrics(stl_mesh):
    """
    Return geometric descriptors used to shape the CV sampling.
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


def _unit_vector(vec, fallback=(1.0, 0.0, 0.0)):
    vec = np.asarray(vec, dtype=float).reshape(3)
    n = float(np.linalg.norm(vec))
    if not np.isfinite(n) or n <= 1e-12:
        vec = np.asarray(fallback, dtype=float)
        n = float(np.linalg.norm(vec))
    return vec / max(n, 1e-12)


def _bounds_mask(points, bounds):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    bounds = np.asarray(bounds, dtype=float)
    return np.all(
        (points >= bounds[:, 0][None, :])
        & (points <= bounds[:, 1][None, :]),
        axis=1,
    )


def _filter_points(
    points,
    stl_mesh,
    bounds,
    epsilon,
    use_signed_distance,
    min_clearance=None,
):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return points
    points = points[_bounds_mask(points, bounds)]
    if len(points) == 0:
        return points
    keep = _mesh_reject_mask(
        points,
        stl_mesh,
        epsilon=epsilon,
        use_signed_distance=use_signed_distance,
    )
    points = points[keep]

    if min_clearance is not None and len(points) > 0:
        clearance = _surface_distance(points, stl_mesh)
        points = points[clearance >= float(min_clearance)]

    return points


def _select_spaced_points(candidates, target_count, min_spacing, rng, existing=None):
    """Greedily keep candidates at least min_spacing away from existing/kept."""
    candidates = np.asarray(candidates, dtype=float).reshape(-1, 3)
    if target_count <= 0 or len(candidates) == 0:
        return np.empty((0, 3), dtype=float)

    if min_spacing is None or min_spacing <= 0.0:
        return candidates[:target_count]

    order = rng.permutation(len(candidates))
    min_dist2 = float(min_spacing) ** 2
    kept = []

    if existing is not None and len(existing) > 0:
        selected = [p for p in np.asarray(existing, dtype=float).reshape(-1, 3)]
    else:
        selected = []

    for idx in order:
        point = candidates[idx]
        if selected:
            selected_arr = np.asarray(selected, dtype=float)
            dist2 = np.sum((selected_arr - point[None, :]) ** 2, axis=1)
            if np.any(dist2 < min_dist2):
                continue

        kept.append(point)
        selected.append(point)
        if len(kept) >= target_count:
            break

    return np.asarray(kept, dtype=float).reshape(-1, 3)


def _minimum_pair_distance(points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) < 2:
        return np.inf
    tree = sp.spatial.cKDTree(points)
    dist, _ = tree.query(points, k=2)
    return float(np.min(dist[:, 1]))


def _allocate_counts(num_samples, base_counts):
    """Scale a nominal sampling split to exactly num_samples."""
    base = np.asarray(base_counts, dtype=float)
    raw = int(num_samples) * base / base.sum()
    counts = np.floor(raw).astype(int)
    remainder = int(num_samples) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        for idx in order[:remainder]:
            counts[idx] += 1
    return counts.tolist()


def _stack_nonempty(parts):
    parts = [p for p in parts if len(p) > 0]
    if not parts:
        return None
    return np.vstack(parts)


def _add_spaced_group(
    selected_parts,
    label_parts,
    candidates,
    target_count,
    label,
    min_spacing,
    rng,
):
    existing = _stack_nonempty(selected_parts)
    pts = _select_spaced_points(
        candidates,
        target_count,
        min_spacing,
        rng,
        existing=existing,
    )
    selected_parts.append(pts)
    label_parts.append(np.full(len(pts), label, dtype=object))
    return pts


def _surface_offset_points(
    stl_mesh,
    num_samples,
    rng,
    offsets,
    bounds,
    epsilon,
    use_signed_distance,
    face_weights=None,
    oversample=8,
    min_clearance=None,
):
    if num_samples <= 0:
        return np.empty((0, 3), dtype=float)

    triangles = np.asarray(stl_mesh.triangles, dtype=float)
    normals = np.asarray(stl_mesh.face_normals, dtype=float)
    areas = np.asarray(stl_mesh.area_faces, dtype=float)

    weights = np.maximum(areas, 0.0)
    if face_weights is not None:
        weights *= np.maximum(np.asarray(face_weights, dtype=float), 0.0)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(triangles), dtype=float)
    probs = weights / weights.sum()

    n_cand = max(num_samples * oversample, num_samples + 20)
    face_idx = rng.choice(len(triangles), size=n_cand, replace=True, p=probs)
    tri = triangles[face_idx]

    u = rng.random(n_cand)
    v = rng.random(n_cand)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]

    surf = (
        tri[:, 0]
        + u[:, None] * (tri[:, 1] - tri[:, 0])
        + v[:, None] * (tri[:, 2] - tri[:, 0])
    )

    face_normals = normals[face_idx]
    offset_values = np.asarray(offsets, dtype=float)
    layer_weights = np.linspace(1.0, 0.4, len(offset_values))
    layer_weights = layer_weights / layer_weights.sum()
    offset_idx = rng.choice(len(offset_values), size=n_cand, p=layer_weights)
    pts = surf + face_normals * offset_values[offset_idx, None]

    try:
        inside = stl_mesh.contains(pts)
        pts[inside] = surf[inside] - face_normals[inside] * offset_values[offset_idx[inside], None]
    except Exception:
        pass

    pts = _filter_points(
        pts,
        stl_mesh,
        bounds,
        epsilon,
        use_signed_distance,
        min_clearance=min_clearance,
    )
    if len(pts) == 0:
        return pts
    return pts[rng.permutation(len(pts))[:num_samples]]


def _face_bias_weights(stl_mesh, wind_direction, kind):
    centers = np.asarray(stl_mesh.triangles_center, dtype=float)
    normals = np.asarray(stl_mesh.face_normals, dtype=float)
    bmin, bmax = np.asarray(stl_mesh.bounds, dtype=float)
    ext = np.maximum(bmax - bmin, 1e-12)
    flow = _unit_vector(wind_direction)

    stream = centers @ flow
    stream_min = float(stream.min())
    stream_span = max(float(stream.max() - stream_min), 1e-12)

    windward = (normals @ flow) < -0.25
    leeward = ((normals @ flow) > 0.25) | (stream > stream_min + 0.65 * stream_span)
    roof = (centers[:, 2] > bmax[2] - 0.18 * ext[2]) | (normals[:, 2] > 0.35)

    near_low = (centers - bmin[None, :]) / ext[None, :] < 0.12
    near_high = (bmax[None, :] - centers) / ext[None, :] < 0.12
    corner_like = (near_low | near_high).sum(axis=1) >= 2

    weights = np.ones(len(centers), dtype=float)
    if kind == "separation":
        weights += 5.0 * leeward + 3.0 * roof + 2.0 * corner_like
    else:
        weights += 4.0 * windward + 3.0 * roof + 2.0 * corner_like
    return weights


def _wake_points(
    stl_mesh,
    num_samples,
    rng,
    bounds,
    epsilon,
    use_signed_distance,
    wind_direction,
    min_clearance=None,
):
    """Randomized space-filling samples in a downstream wake envelope."""
    if num_samples <= 0:
        return np.empty((0, 3), dtype=float)

    bm = _body_metrics(stl_mesh)
    bmin, bmax = bm["bmin"], bm["bmax"]
    ext = np.maximum(bmax - bmin, 1e-12)
    center = 0.5 * (bmin + bmax)
    flow = _unit_vector(wind_direction)
    axis = int(np.argmax(np.abs(flow)))
    sign = 1.0 if flow[axis] >= 0.0 else -1.0
    cross_axes = [ax for ax in range(3) if ax != axis]
    diameter = max(float(ext[cross_axes[0]]), float(ext[cross_axes[1]]))

    start = max(0.25 * diameter, float(min_clearance or 0.0), 4.0 * epsilon)
    length = 6.0 * diameter
    if sign > 0.0:
        downstream_space = bounds[axis, 1] - bmax[axis]
    else:
        downstream_space = bmin[axis] - bounds[axis, 0]
    length = min(length, max(downstream_space - start, 0.5 * diameter))

    parts = []
    collected = 0
    for _ in range(12):
        if collected >= num_samples:
            break

        n_cand = max((num_samples - collected) * 35, 300)
        pts = np.empty((n_cand, 3), dtype=float)

        # Continuous streamwise positions, biased toward the near wake but
        # with some points kept farther downstream for recovery shape.
        near_mask = rng.random(n_cand) < 0.70
        stream_dist = np.empty(n_cand, dtype=float)
        stream_dist[near_mask] = start + rng.beta(1.2, 3.2, near_mask.sum()) * length
        stream_dist[~near_mask] = start + rng.random((~near_mask).sum()) * length

        if sign > 0.0:
            pts[:, axis] = bmax[axis] + stream_dist
        else:
            pts[:, axis] = bmin[axis] - stream_dist

        s_norm = np.clip((stream_dist - start) / max(length, 1e-12), 0.0, 1.0)
        base_width = np.array(
            [
                max(0.55 * ext[cross_axes[0]], 0.35 * diameter),
                max(0.50 * ext[cross_axes[1]], 0.35 * diameter),
            ],
            dtype=float,
        )
        growth_width = np.array(
            [
                max(0.75 * ext[cross_axes[0]], 0.45 * diameter),
                max(0.65 * ext[cross_axes[1]], 0.40 * diameter),
            ],
            dtype=float,
        )
        half_width = base_width[None, :] + s_norm[:, None] * growth_width[None, :]

        # Mixture: most points near the wake core/shear layer, some across the
        # wider envelope so the GP does not overfit a thin line.
        core = rng.random(n_cand) < 0.65
        cross = np.empty((n_cand, 2), dtype=float)
        cross[core] = rng.normal(0.0, 0.45, size=(core.sum(), 2))
        cross[~core] = rng.uniform(-1.0, 1.0, size=((~core).sum(), 2))
        cross = np.clip(cross, -1.0, 1.0)

        pts[:, cross_axes] = center[cross_axes][None, :] + cross * half_width

        pts = _filter_points(
            pts,
            stl_mesh,
            bounds,
            epsilon,
            use_signed_distance,
            min_clearance=min_clearance,
        )
        if len(pts) > 0:
            parts.append(pts)
            collected += len(pts)

    if not parts:
        return np.empty((0, 3), dtype=float)

    pts = np.vstack(parts)
    pts = pts[rng.permutation(len(pts))]
    return pts[:num_samples]


def _wake_envelope_points(
    stl_mesh,
    num_samples,
    rng,
    bounds,
    epsilon,
    use_signed_distance,
    wind_direction,
    min_clearance=None,
    stream_fraction=(0.0, 1.0),
    cross_mode="core",
    length_factor=7.5,
):
    """Unstructured samples in a controllable downstream wake subregion."""
    if num_samples <= 0:
        return np.empty((0, 3), dtype=float)

    bm = _body_metrics(stl_mesh)
    bmin, bmax = bm["bmin"], bm["bmax"]
    ext = np.maximum(bmax - bmin, 1e-12)
    center = 0.5 * (bmin + bmax)
    flow = _unit_vector(wind_direction)
    axis = int(np.argmax(np.abs(flow)))
    sign = 1.0 if flow[axis] >= 0.0 else -1.0
    cross_axes = [ax for ax in range(3) if ax != axis]
    diameter = max(float(ext[cross_axes[0]]), float(ext[cross_axes[1]]))

    start = max(0.25 * diameter, float(min_clearance or 0.0), 4.0 * epsilon)
    length = float(length_factor) * diameter
    if sign > 0.0:
        downstream_space = bounds[axis, 1] - bmax[axis]
    else:
        downstream_space = bmin[axis] - bounds[axis, 0]
    length = min(length, max(downstream_space - start, 0.5 * diameter))
    f0, f1 = np.clip(np.asarray(stream_fraction, dtype=float), 0.0, 1.0)
    if f1 <= f0:
        f0, f1 = 0.0, 1.0

    parts = []
    collected = 0
    for _ in range(10):
        if collected >= num_samples:
            break

        n_cand = max((num_samples - collected) * 18, 240)
        pts = np.empty((n_cand, 3), dtype=float)

        t = rng.random(n_cand)
        if cross_mode == "core":
            t = rng.beta(1.15, 2.8, n_cand)
        elif cross_mode == "recovery":
            t = rng.beta(1.8, 1.2, n_cand)
        stream_dist = start + (f0 + t * (f1 - f0)) * length

        if sign > 0.0:
            pts[:, axis] = bmax[axis] + stream_dist
        else:
            pts[:, axis] = bmin[axis] - stream_dist

        s_norm = np.clip((stream_dist - start) / max(length, 1e-12), 0.0, 1.0)
        base_width = np.array(
            [
                max(0.45 * ext[cross_axes[0]], 0.30 * diameter),
                max(0.42 * ext[cross_axes[1]], 0.28 * diameter),
            ],
            dtype=float,
        )
        growth_width = np.array(
            [
                max(0.85 * ext[cross_axes[0]], 0.55 * diameter),
                max(0.72 * ext[cross_axes[1]], 0.45 * diameter),
            ],
            dtype=float,
        )
        half_width = base_width[None, :] + s_norm[:, None] * growth_width[None, :]

        if cross_mode == "core":
            cross = rng.normal(0.0, 0.28, size=(n_cand, 2))
            cross = np.clip(cross, -0.75, 0.75)
        elif cross_mode == "shear":
            theta = rng.uniform(0.0, 2.0 * np.pi, n_cand)
            radius = rng.uniform(0.45, 1.05, n_cand)
            cross = np.column_stack([np.cos(theta), np.sin(theta)]) * radius[:, None]
        elif cross_mode == "recovery":
            cross = rng.uniform(-1.0, 1.0, size=(n_cand, 2))
        else:
            core = rng.random(n_cand) < 0.60
            cross = np.empty((n_cand, 2), dtype=float)
            cross[core] = rng.normal(0.0, 0.35, size=(core.sum(), 2))
            cross[~core] = rng.uniform(-1.0, 1.0, size=((~core).sum(), 2))
            cross = np.clip(cross, -1.0, 1.0)

        pts[:, cross_axes] = center[cross_axes][None, :] + cross * half_width
        pts = _filter_points(
            pts,
            stl_mesh,
            bounds,
            epsilon,
            use_signed_distance,
            min_clearance=min_clearance,
        )
        if len(pts) > 0:
            parts.append(pts)
            collected += len(pts)

    if not parts:
        return np.empty((0, 3), dtype=float)

    pts = np.vstack(parts)
    pts = pts[rng.permutation(len(pts))]
    return pts[:num_samples]


def _upstream_anchor_points(
    stl_mesh,
    num_samples,
    rng,
    bounds,
    epsilon,
    use_signed_distance,
    wind_direction,
    min_clearance=None,
):
    if num_samples <= 0:
        return np.empty((0, 3), dtype=float)

    bm = _body_metrics(stl_mesh)
    bmin, bmax = bm["bmin"], bm["bmax"]
    ext = np.maximum(bmax - bmin, 1e-12)
    center = 0.5 * (bmin + bmax)
    flow = _unit_vector(wind_direction)
    axis = int(np.argmax(np.abs(flow)))
    sign = 1.0 if flow[axis] >= 0.0 else -1.0
    cross_axes = [ax for ax in range(3) if ax != axis]

    lo = bounds[:, 0].copy()
    hi = bounds[:, 1].copy()
    if sign > 0.0:
        hi[axis] = min(hi[axis], bmin[axis] - 0.5 * ext[axis])
    else:
        lo[axis] = max(lo[axis], bmax[axis] + 0.5 * ext[axis])

    for ax in cross_axes:
        span = max(1.25 * ext[ax], 8.0 * epsilon)
        lo[ax] = max(lo[ax], center[ax] - span)
        hi[ax] = min(hi[ax], center[ax] + span)

    if np.any(hi <= lo):
        return np.empty((0, 3), dtype=float)

    pts = rng.uniform(lo[None, :], hi[None, :], size=(num_samples * 8, 3))
    pts = _filter_points(
        pts,
        stl_mesh,
        bounds,
        epsilon,
        use_signed_distance,
        min_clearance=min_clearance,
    )
    if len(pts) == 0:
        return pts
    return pts[:num_samples]


def _drone_safe_fill_points(
    stl_mesh,
    num_samples,
    rng,
    bounds,
    epsilon,
    use_signed_distance,
    wind_direction,
    min_clearance,
):
    """Fallback points in a drone-safe box around the body and near wake."""
    if num_samples <= 0:
        return np.empty((0, 3), dtype=float)

    bm = _body_metrics(stl_mesh)
    bmin, bmax = bm["bmin"], bm["bmax"]
    ext = np.maximum(bmax - bmin, 1e-12)
    center = 0.5 * (bmin + bmax)

    flow = _unit_vector(wind_direction)
    axis = int(np.argmax(np.abs(flow)))
    sign = 1.0 if flow[axis] >= 0.0 else -1.0
    cross_axes = [ax for ax in range(3) if ax != axis]
    diameter = max(float(ext[cross_axes[0]]), float(ext[cross_axes[1]]))

    lo = bounds[:, 0].copy()
    hi = bounds[:, 1].copy()

    if sign > 0.0:
        lo[axis] = max(lo[axis], bmin[axis] - 4.0 * diameter)
        hi[axis] = min(hi[axis], bmax[axis] + 5.0 * diameter)
    else:
        lo[axis] = max(lo[axis], bmin[axis] - 5.0 * diameter)
        hi[axis] = min(hi[axis], bmax[axis] + 4.0 * diameter)

    for ax in cross_axes:
        half_width = max(1.6 * ext[ax], min_clearance)
        lo[ax] = max(lo[ax], center[ax] - half_width)
        hi[ax] = min(hi[ax], center[ax] + half_width)

    if np.any(hi <= lo):
        lo = bounds[:, 0].copy()
        hi = bounds[:, 1].copy()

    parts = []
    collected = 0
    for _ in range(12):
        if collected >= num_samples:
            break
        n_cand = max((num_samples - collected) * 40, 200)
        cand = rng.uniform(lo[None, :], hi[None, :], size=(n_cand, 3))
        cand = _filter_points(
            cand,
            stl_mesh,
            bounds,
            epsilon,
            use_signed_distance,
            min_clearance=min_clearance,
        )
        if len(cand) > 0:
            parts.append(cand)
            collected += len(cand)

    if not parts:
        return np.empty((0, 3), dtype=float)

    pts = np.vstack(parts)
    pts = pts[rng.permutation(len(pts))]
    return pts[:num_samples]


def _domain_spaced_fill_points(
    stl_mesh,
    num_samples,
    rng,
    bounds,
    epsilon,
    use_signed_distance,
    min_clearance,
    min_spacing,
):
    """Sparse CFD-domain lattice fill that respects drone clearance."""
    if num_samples <= 0:
        return np.empty((0, 3), dtype=float)

    bounds = np.asarray(bounds, dtype=float)
    step = max(float(min_spacing) * 1.05, 1e-12)
    candidates = []

    for _ in range(8):
        phase = rng.uniform(0.15 * step, 0.85 * step, size=3)
        axes = [
            np.arange(bounds[i, 0] + phase[i], bounds[i, 1] + 1e-9, step)
            for i in range(3)
        ]
        if any(len(axis) == 0 for axis in axes):
            continue
        x, y, z = np.meshgrid(*axes, indexing="ij")
        grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        candidates.append(grid)

    if not candidates:
        return np.empty((0, 3), dtype=float)

    pts = np.vstack(candidates)
    pts = pts[rng.permutation(len(pts))]
    pts = _filter_points(
        pts,
        stl_mesh,
        bounds,
        epsilon,
        use_signed_distance,
        min_clearance=min_clearance,
    )
    if len(pts) == 0:
        return pts
    return pts[: max(num_samples, 1)]


def _control_volume_bounds(stl_mesh, bounds, wind_direction, min_clearance):
    bm = _body_metrics(stl_mesh)
    bmin, bmax = bm["bmin"], bm["bmax"]
    ext = np.maximum(bmax - bmin, 1e-12)
    flow = _unit_vector(wind_direction)
    axis = int(np.argmax(np.abs(flow)))
    sign = 1.0 if flow[axis] >= 0.0 else -1.0
    cross_axes = [ax for ax in range(3) if ax != axis]
    diameter = max(float(ext[cross_axes[0]]), float(ext[cross_axes[1]]))

    cross_pad = np.maximum(0.22 * ext, float(min_clearance))
    upstream_pad = max(float(min_clearance), 0.35 * diameter)
    downstream_pad = max(2.0 * float(min_clearance), 0.95 * diameter)

    cv_lo = bmin - cross_pad
    cv_hi = bmax + cross_pad
    if sign > 0.0:
        cv_lo[axis] = bmin[axis] - upstream_pad
        cv_hi[axis] = bmax[axis] + downstream_pad
    else:
        cv_lo[axis] = bmin[axis] - downstream_pad
        cv_hi[axis] = bmax[axis] + upstream_pad

    bounds = np.asarray(bounds, dtype=float)
    cv_lo = np.maximum(cv_lo, bounds[:, 0])
    cv_hi = np.minimum(cv_hi, bounds[:, 1])
    return cv_lo, cv_hi, axis, sign, cross_axes


def _control_volume_face_points(
    stl_mesh,
    num_samples,
    rng,
    bounds,
    epsilon,
    use_signed_distance,
    wind_direction,
    min_clearance,
    face_role,
):
    """Random points on a closed box-like control volume around the body."""
    if num_samples <= 0:
        return np.empty((0, 3), dtype=float)

    cv_lo, cv_hi, axis, sign, cross_axes = _control_volume_bounds(
        stl_mesh, bounds, wind_direction, min_clearance
    )
    if np.any(cv_hi <= cv_lo):
        return np.empty((0, 3), dtype=float)

    upstream_face = cv_lo[axis] if sign > 0.0 else cv_hi[axis]
    downstream_face = cv_hi[axis] if sign > 0.0 else cv_lo[axis]

    if face_role == "windward":
        faces = [(axis, upstream_face)]
    elif face_role == "leeward":
        faces = [(axis, downstream_face)]
    else:
        faces = []
        for ax in cross_axes:
            for value in (cv_lo[ax], cv_hi[ax]):
                if ax == 2 and value <= bounds[2, 0] + 1e-9:
                    continue
                faces.append((ax, value))

    if not faces:
        return np.empty((0, 3), dtype=float)

    lengths = cv_hi - cv_lo
    areas = []
    for face_axis, _ in faces:
        other_axes = [ax for ax in range(3) if ax != face_axis]
        areas.append(float(np.prod(lengths[other_axes])))
    probs = np.asarray(areas, dtype=float)
    if probs.sum() <= 0.0:
        probs = np.ones(len(faces), dtype=float)
    probs = probs / probs.sum()

    parts = []
    collected = 0
    for _ in range(10):
        if collected >= num_samples:
            break

        n_cand = max((num_samples - collected) * 14, 160)
        pts = rng.uniform(cv_lo[None, :], cv_hi[None, :], size=(n_cand, 3))
        face_idx = rng.choice(len(faces), size=n_cand, replace=True, p=probs)
        for idx, (face_axis, value) in enumerate(faces):
            mask = face_idx == idx
            pts[mask, face_axis] = value

        pts = _filter_points(
            pts,
            stl_mesh,
            bounds,
            epsilon,
            use_signed_distance,
            min_clearance=min_clearance,
        )
        if len(pts) > 0:
            parts.append(pts)
            collected += len(pts)

    if not parts:
        return np.empty((0, 3), dtype=float)

    pts = np.vstack(parts)
    pts = pts[rng.permutation(len(pts))]
    return pts[:num_samples]


def _wake_rmse_points(
    stl_mesh,
    num_samples,
    epsilon,
    use_signed_distance,
    rng,
    bounds,
    wind_direction=(1.0, 0.0, 0.0),
    shell_offsets=(0.50, 0.75, 1.00),
    min_drone_clearance=0.50,
    min_measurement_spacing=0.50,
):
    """
    Velocity-field sampler: mostly unstructured wake volume, with enough
    near-body and freestream anchors to keep the GP from inventing structure.
    """
    n_core, n_shear, n_recovery, n_near, n_far = _allocate_counts(
        num_samples, [42, 34, 18, 16, 10]
    )
    selected_parts = []
    label_parts = []
    candidate_multiplier = 8

    core_candidates = _wake_envelope_points(
        stl_mesh,
        n_core * candidate_multiplier,
        rng,
        bounds,
        epsilon,
        use_signed_distance,
        wind_direction,
        min_clearance=min_drone_clearance,
        stream_fraction=(0.0, 0.72),
        cross_mode="core",
        length_factor=7.5,
    )
    core = _add_spaced_group(
        selected_parts, label_parts, core_candidates, n_core,
        "wake core", min_measurement_spacing, rng,
    )

    shear_candidates = _wake_envelope_points(
        stl_mesh,
        n_shear * candidate_multiplier,
        rng,
        bounds,
        epsilon,
        use_signed_distance,
        wind_direction,
        min_clearance=min_drone_clearance,
        stream_fraction=(0.0, 0.82),
        cross_mode="shear",
        length_factor=7.5,
    )
    shear = _add_spaced_group(
        selected_parts, label_parts, shear_candidates, n_shear,
        "wake shear", min_measurement_spacing, rng,
    )

    recovery_candidates = _wake_envelope_points(
        stl_mesh,
        n_recovery * candidate_multiplier,
        rng,
        bounds,
        epsilon,
        use_signed_distance,
        wind_direction,
        min_clearance=min_drone_clearance,
        stream_fraction=(0.55, 1.0),
        cross_mode="recovery",
        length_factor=7.5,
    )
    recovery = _add_spaced_group(
        selected_parts, label_parts, recovery_candidates, n_recovery,
        "wake recovery", min_measurement_spacing, rng,
    )

    near_candidates = _surface_offset_points(
        stl_mesh,
        n_near * candidate_multiplier,
        rng,
        shell_offsets,
        bounds,
        epsilon,
        use_signed_distance,
        face_weights=_face_bias_weights(stl_mesh, wind_direction, "separation"),
        oversample=8,
        min_clearance=min_drone_clearance,
    )
    near = _add_spaced_group(
        selected_parts, label_parts, near_candidates, n_near,
        "near-body guards", min_measurement_spacing, rng,
    )

    far_candidates = _upstream_anchor_points(
        stl_mesh,
        n_far * candidate_multiplier,
        rng,
        bounds,
        epsilon,
        use_signed_distance,
        wind_direction,
        min_clearance=min_drone_clearance,
    )
    far = _add_spaced_group(
        selected_parts, label_parts, far_candidates, n_far,
        "freestream anchors", min_measurement_spacing, rng,
    )

    pts = _stack_nonempty(selected_parts)
    if pts is None:
        pts = np.empty((0, 3), dtype=float)
    labels = (
        np.concatenate([lbl for lbl in label_parts if len(lbl) > 0])
        if any(len(lbl) > 0 for lbl in label_parts)
        else np.empty(0, dtype=object)
    )

    if len(pts) < num_samples:
        fill_candidates = _drone_safe_fill_points(
            stl_mesh,
            max((num_samples - len(pts)) * 30, 240),
            rng,
            bounds,
            epsilon,
            use_signed_distance,
            wind_direction,
            min_drone_clearance,
        )
        fill = _select_spaced_points(
            fill_candidates,
            num_samples - len(pts),
            min_measurement_spacing,
            rng,
            existing=pts,
        )
        if len(fill) > 0:
            pts = np.vstack([pts, fill])
            labels = np.concatenate(
                [labels, np.full(len(fill), "domain guards", dtype=object)]
            )

    if len(pts) > 0:
        order = rng.permutation(len(pts))
        pts = pts[order]
        labels = labels[order]

    print("\nWake-RMSE sampling")
    print("------------------")
    print(f"Requested samples:       {num_samples}")
    print(f"Wake core target:        {n_core}  returned {len(core)}")
    print(f"Wake shear target:       {n_shear}  returned {len(shear)}")
    print(f"Wake recovery target:    {n_recovery}  returned {len(recovery)}")
    print(f"Near-body guard target:  {n_near}  returned {len(near)}")
    print(f"Freestream target:       {n_far}  returned {len(far)}")
    print(f"Minimum clearance:       {min_drone_clearance}")
    print(f"Minimum point spacing:   {min_measurement_spacing}")
    print(f"Achieved point spacing:  {_minimum_pair_distance(pts):.6g}")
    print(f"Returning:               {min(len(pts), num_samples)}")

    return pts[:num_samples], labels[:num_samples]


def _force_cv_points(
    stl_mesh,
    num_samples,
    epsilon,
    use_signed_distance,
    rng,
    bounds,
    wind_direction=(1.0, 0.0, 0.0),
    shell_offsets=(0.50, 0.75, 1.00),
    min_drone_clearance=0.50,
    min_measurement_spacing=0.50,
):
    """
    Force sampler: dense measurements on a near-body control volume, plus a
    small number of separation and inflow anchors to stabilize the GP.
    """
    n_in, n_out, n_side, n_sep, n_far = _allocate_counts(
        num_samples, [30, 30, 36, 14, 10]
    )
    selected_parts = []
    label_parts = []
    candidate_multiplier = 8

    in_candidates = _control_volume_face_points(
        stl_mesh,
        n_in * candidate_multiplier,
        rng,
        bounds,
        epsilon,
        use_signed_distance,
        wind_direction,
        min_drone_clearance,
        face_role="windward",
    )
    inflow = _add_spaced_group(
        selected_parts, label_parts, in_candidates, n_in,
        "cv windward face", min_measurement_spacing, rng,
    )

    out_candidates = _control_volume_face_points(
        stl_mesh,
        n_out * candidate_multiplier,
        rng,
        bounds,
        epsilon,
        use_signed_distance,
        wind_direction,
        min_drone_clearance,
        face_role="leeward",
    )
    outflow = _add_spaced_group(
        selected_parts, label_parts, out_candidates, n_out,
        "cv leeward face", min_measurement_spacing, rng,
    )

    side_candidates = _control_volume_face_points(
        stl_mesh,
        n_side * candidate_multiplier,
        rng,
        bounds,
        epsilon,
        use_signed_distance,
        wind_direction,
        min_drone_clearance,
        face_role="side_top",
    )
    side = _add_spaced_group(
        selected_parts, label_parts, side_candidates, n_side,
        "cv side/top faces", min_measurement_spacing, rng,
    )

    sep_candidates = _wake_envelope_points(
        stl_mesh,
        n_sep * candidate_multiplier,
        rng,
        bounds,
        epsilon,
        use_signed_distance,
        wind_direction,
        min_clearance=min_drone_clearance,
        stream_fraction=(0.0, 0.25),
        cross_mode="shear",
        length_factor=3.5,
    )
    sep = _add_spaced_group(
        selected_parts, label_parts, sep_candidates, n_sep,
        "separation guards", min_measurement_spacing, rng,
    )

    far_candidates = _upstream_anchor_points(
        stl_mesh,
        n_far * candidate_multiplier,
        rng,
        bounds,
        epsilon,
        use_signed_distance,
        wind_direction,
        min_clearance=min_drone_clearance,
    )
    far = _add_spaced_group(
        selected_parts, label_parts, far_candidates, n_far,
        "upstream anchors", min_measurement_spacing, rng,
    )

    pts = _stack_nonempty(selected_parts)
    if pts is None:
        pts = np.empty((0, 3), dtype=float)
    labels = (
        np.concatenate([lbl for lbl in label_parts if len(lbl) > 0])
        if any(len(lbl) > 0 for lbl in label_parts)
        else np.empty(0, dtype=object)
    )

    if len(pts) < num_samples:
        fill_candidates = _drone_safe_fill_points(
            stl_mesh,
            max((num_samples - len(pts)) * 30, 240),
            rng,
            bounds,
            epsilon,
            use_signed_distance,
            wind_direction,
            min_drone_clearance,
        )
        fill = _select_spaced_points(
            fill_candidates,
            num_samples - len(pts),
            min_measurement_spacing,
            rng,
            existing=pts,
        )
        if len(fill) > 0:
            pts = np.vstack([pts, fill])
            labels = np.concatenate(
                [labels, np.full(len(fill), "domain guards", dtype=object)]
            )

    if len(pts) > 0:
        order = rng.permutation(len(pts))
        pts = pts[order]
        labels = labels[order]

    print("\nForce-CV sampling")
    print("-----------------")
    print(f"Requested samples:       {num_samples}")
    print(f"CV windward target:      {n_in}  returned {len(inflow)}")
    print(f"CV leeward target:       {n_out}  returned {len(outflow)}")
    print(f"CV side/top target:      {n_side}  returned {len(side)}")
    print(f"Separation guard target: {n_sep}  returned {len(sep)}")
    print(f"Upstream target:         {n_far}  returned {len(far)}")
    print(f"Shell offsets:           {np.asarray(shell_offsets, dtype=float)}")
    print(f"Minimum clearance:       {min_drone_clearance}")
    print(f"Minimum point spacing:   {min_measurement_spacing}")
    print(f"Achieved point spacing:  {_minimum_pair_distance(pts):.6g}")
    print(f"Returning:               {min(len(pts), num_samples)}")

    return pts[:num_samples], labels[:num_samples]


def _force_wake_points(
    stl_mesh,
    num_samples,
    epsilon,
    use_signed_distance,
    rng,
    bounds,
    wind_direction=(1.0, 0.0, 0.0),
    shell_offsets=(0.50, 0.75, 1.00),
    min_drone_clearance=0.50,
    min_measurement_spacing=0.50,
):
    """
    Force-first, wake-aware sampler. For 120 points the target split is:
    46 separation/rear shell, 40 wake, 24 windward/edge, 10 upstream.
    The default shell is kept at least 0.5 m from the building to represent
    drone-safe measurement standoff rather than wall-adjacent probes.
    """
    num_samples = int(num_samples)
    base = np.array([46, 40, 24, 10], dtype=float)
    raw = num_samples * base / base.sum()
    counts = np.floor(raw).astype(int)
    for idx in np.argsort(raw - counts)[::-1][: num_samples - counts.sum()]:
        counts[idx] += 1
    n_sep, n_wake, n_bias, n_far = counts.tolist()

    candidate_multiplier = 10
    selected_parts = []
    label_parts = []

    sep_candidates = _surface_offset_points(
        stl_mesh,
        n_sep * candidate_multiplier,
        rng,
        shell_offsets,
        bounds,
        epsilon,
        use_signed_distance,
        face_weights=_face_bias_weights(stl_mesh, wind_direction, "separation"),
        oversample=10,
        min_clearance=min_drone_clearance,
    )
    sep = _select_spaced_points(
        sep_candidates,
        n_sep,
        min_measurement_spacing,
        rng,
    )
    selected_parts.append(sep)
    label_parts.append(np.full(len(sep), "separation shell", dtype=object))

    wake_candidates = _wake_points(
        stl_mesh,
        n_wake * candidate_multiplier,
        rng,
        bounds,
        epsilon,
        use_signed_distance,
        wind_direction,
        min_clearance=min_drone_clearance,
    )
    wake = _select_spaced_points(
        wake_candidates,
        n_wake,
        min_measurement_spacing,
        rng,
        existing=np.vstack([p for p in selected_parts if len(p) > 0])
        if any(len(p) > 0 for p in selected_parts)
        else None,
    )
    selected_parts.append(wake)
    label_parts.append(np.full(len(wake), "wake", dtype=object))

    bias_candidates = _surface_offset_points(
        stl_mesh,
        n_bias * candidate_multiplier,
        rng,
        shell_offsets,
        bounds,
        epsilon,
        use_signed_distance,
        face_weights=_face_bias_weights(stl_mesh, wind_direction, "windward"),
        oversample=10,
        min_clearance=min_drone_clearance,
    )
    bias = _select_spaced_points(
        bias_candidates,
        n_bias,
        min_measurement_spacing,
        rng,
        existing=np.vstack([p for p in selected_parts if len(p) > 0])
        if any(len(p) > 0 for p in selected_parts)
        else None,
    )
    selected_parts.append(bias)
    label_parts.append(np.full(len(bias), "windward/edge", dtype=object))

    far_candidates = _upstream_anchor_points(
        stl_mesh,
        n_far * candidate_multiplier,
        rng,
        bounds,
        epsilon,
        use_signed_distance,
        wind_direction,
        min_clearance=min_drone_clearance,
    )
    far = _select_spaced_points(
        far_candidates,
        n_far,
        min_measurement_spacing,
        rng,
        existing=np.vstack([p for p in selected_parts if len(p) > 0])
        if any(len(p) > 0 for p in selected_parts)
        else None,
    )
    selected_parts.append(far)
    label_parts.append(np.full(len(far), "upstream", dtype=object))

    parts = [p for p in selected_parts if len(p) > 0]
    pts = np.vstack(parts) if parts else np.empty((0, 3), dtype=float)
    labels = (
        np.concatenate([lbl for lbl in label_parts if len(lbl) > 0])
        if any(len(lbl) > 0 for lbl in label_parts)
        else np.empty(0, dtype=object)
    )

    if len(pts) < num_samples:
        fill_candidates = _wake_points(
            stl_mesh,
            max((num_samples - len(pts)) * 30, 200),
            rng,
            bounds,
            epsilon,
            use_signed_distance,
            wind_direction,
            min_clearance=min_drone_clearance,
        )
        fill = _select_spaced_points(
            fill_candidates,
            num_samples - len(pts),
            min_measurement_spacing,
            rng,
            existing=pts,
        )
        if len(fill) > 0:
            pts = np.vstack([pts, fill])
            labels = np.concatenate(
                [labels, np.full(len(fill), "wake fill", dtype=object)]
            )

    if len(pts) < num_samples:
        fill_candidates = _drone_safe_fill_points(
            stl_mesh,
            max((num_samples - len(pts)) * 50, 300),
            rng,
            bounds,
            epsilon,
            use_signed_distance,
            wind_direction,
            min_drone_clearance,
        )
        fill = _select_spaced_points(
            fill_candidates,
            num_samples - len(pts),
            min_measurement_spacing,
            rng,
            existing=pts,
        )
        if len(fill) > 0:
            pts = np.vstack([pts, fill])
            labels = np.concatenate(
                [labels, np.full(len(fill), "drone-safe fill", dtype=object)]
            )

    if len(pts) < num_samples:
        fill_candidates = _domain_spaced_fill_points(
            stl_mesh,
            max((num_samples - len(pts)) * 50, 500),
            rng,
            bounds,
            epsilon,
            use_signed_distance,
            min_drone_clearance,
            min_measurement_spacing,
        )
        fill = _select_spaced_points(
            fill_candidates,
            num_samples - len(pts),
            min_measurement_spacing,
            rng,
            existing=pts,
        )
        if len(fill) > 0:
            pts = np.vstack([pts, fill])
            labels = np.concatenate(
                [labels, np.full(len(fill), "domain fill", dtype=object)]
            )

    if len(pts) > 0:
        order = rng.permutation(len(pts))
        pts = pts[order]
        labels = labels[order]

    print("\nForce-wake sampling")
    print("-------------------")
    print(f"Requested samples:       {num_samples}")
    print(f"Separation shell target: {n_sep}  returned {len(sep)}")
    print(f"Wake target:             {n_wake}  returned {len(wake)}")
    print(f"Windward/edge target:    {n_bias}  returned {len(bias)}")
    print(f"Upstream target:         {n_far}  returned {len(far)}")
    print(f"Shell offsets:           {np.asarray(shell_offsets, dtype=float)}")
    print(f"Minimum clearance:       {min_drone_clearance}")
    print(f"Minimum point spacing:   {min_measurement_spacing}")
    print(f"Achieved point spacing:  {_minimum_pair_distance(pts):.6g}")
    print(f"Returning:               {min(len(pts), num_samples)}")

    return pts[:num_samples], labels[:num_samples]


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
    z_max=7.5,
    oversample=6,
):
    """
    Sample points on the lateral wall of a vertical cylinder centred at
    (x=0, y=0), spanning z in [z_min, z_max].
    """
    bm = _body_metrics(stl_mesh)

    if cylinder_radius is None:
        half_x = 0.5 * bm["length_x"]
        half_y = 0.5 * bm["width_y"]
        cylinder_radius = 1.5 * np.hypot(half_x, half_y)
        cylinder_radius = max(cylinder_radius, bm["diameter"])

    r = float(cylinder_radius)
    n_cand = num_samples * oversample

    n_bands = max(10, num_samples // 5)
    z_edges = np.linspace(z_min, z_max, n_bands + 1)
    per_band = n_cand // n_bands
    remainder = n_cand - per_band * n_bands

    z_cand_parts = []
    for i in range(n_bands):
        n_here = per_band + (1 if i < remainder else 0)
        z_cand_parts.append(rng.uniform(z_edges[i], z_edges[i + 1], n_here))
    z_cand = np.concatenate(z_cand_parts)

    phi = rng.uniform(0.0, 2.0 * np.pi, len(z_cand))

    x_cand = r * np.cos(phi)
    y_cand = r * np.sin(phi)
    pts = np.column_stack([x_cand, y_cand, z_cand])

    keep = _mesh_reject_mask(
        pts, stl_mesh, epsilon=epsilon, use_signed_distance=use_signed_distance
    )
    pts = pts[keep]

    idx = rng.permutation(len(pts))
    pts = pts[idx]

    print("\nCylinder-wall sampling (CV method)")
    print("------------------------------------")
    print(f"Cylinder radius:  {r:.4g}")
    print(f"Axial range:      z = {z_min:.3g} to {z_max:.3g}")
    print(f"Candidates:       {n_cand}  (oversample x{oversample})")
    print(f"After rejection:  {len(pts)}")
    print(f"Returning:        {min(len(pts), num_samples)}")

    return pts[:num_samples]


def _named_strategy_points(
    method,
    stl_mesh,
    point_cap,
    bounds,
    epsilon,
    use_signed_distance,
    rng,
    cylinder_radius=None,
    cylinder_z_min=0.0,
    cylinder_z_max=7.5,
    wind_direction=(1.0, 0.0, 0.0),
    shell_offsets=(0.50, 0.75, 1.00),
    min_drone_clearance=0.50,
    min_measurement_spacing=0.50,
):
    sample_groups = None

    if method == "cv":
        valid_points = _cylinder_wall_points(
            stl_mesh=stl_mesh,
            num_samples=point_cap,
            epsilon=epsilon,
            use_signed_distance=use_signed_distance,
            rng=rng,
            cylinder_radius=cylinder_radius,
            z_min=cylinder_z_min,
            z_max=cylinder_z_max,
        )
    elif method == "wake_rmse":
        valid_points, sample_groups = _wake_rmse_points(
            stl_mesh=stl_mesh,
            num_samples=point_cap,
            epsilon=epsilon,
            use_signed_distance=use_signed_distance,
            rng=rng,
            bounds=bounds,
            wind_direction=wind_direction,
            shell_offsets=shell_offsets,
            min_drone_clearance=min_drone_clearance,
            min_measurement_spacing=min_measurement_spacing,
        )
    elif method == "force_cv":
        valid_points, sample_groups = _force_cv_points(
            stl_mesh=stl_mesh,
            num_samples=point_cap,
            epsilon=epsilon,
            use_signed_distance=use_signed_distance,
            rng=rng,
            bounds=bounds,
            wind_direction=wind_direction,
            shell_offsets=shell_offsets,
            min_drone_clearance=min_drone_clearance,
            min_measurement_spacing=min_measurement_spacing,
        )
    elif method == "force_wake":
        valid_points, sample_groups = _force_wake_points(
            stl_mesh=stl_mesh,
            num_samples=point_cap,
            epsilon=epsilon,
            use_signed_distance=use_signed_distance,
            rng=rng,
            bounds=bounds,
            wind_direction=wind_direction,
            shell_offsets=shell_offsets,
            min_drone_clearance=min_drone_clearance,
            min_measurement_spacing=min_measurement_spacing,
        )
    else:
        raise ValueError(f"Unknown named sampling method: {method}")

    if len(valid_points) == 0:
        raise RuntimeError(
            f"{method} sampling produced no valid points. "
            "Check mesh/CFD units, clearance, spacing, and CFD bounds."
        )

    return valid_points, sample_groups


def cfd_bounds_from_file(field_path):
    cfd_df = pd.read_csv(field_path)
    cfd_df.columns = cfd_df.columns.str.strip()
    source_coords = cfd_df[
        ["x-coordinate", "y-coordinate", "z-coordinate"]
    ].values.astype(float)
    return np.array(
        [
            [source_coords[:, 0].min(), source_coords[:, 0].max()],
            [source_coords[:, 1].min(), source_coords[:, 1].max()],
            [source_coords[:, 2].min(), source_coords[:, 2].max()],
        ],
        dtype=float,
    )


def preview_sampling_points(
    field_path,
    stl_mesh,
    method,
    num_samples=120,
    epsilon=0.02,
    use_signed_distance=True,
    max_points=120,
    cylinder_radius=None,
    cylinder_z_min=0.0,
    cylinder_z_max=7.5,
    wind_direction=(1.0, 0.0, 0.0),
    shell_offsets=(0.50, 0.75, 1.00),
    min_drone_clearance=0.50,
    min_measurement_spacing=0.50,
):
    """Generate sampling coordinates and labels without CFD interpolation."""
    bounds = cfd_bounds_from_file(field_path)
    point_cap = int(min(num_samples, max_points))
    rng = np.random.default_rng(seed)
    valid_points, sample_groups = _named_strategy_points(
        method,
        stl_mesh,
        point_cap,
        bounds,
        epsilon,
        use_signed_distance,
        rng,
        cylinder_radius=cylinder_radius,
        cylinder_z_min=cylinder_z_min,
        cylinder_z_max=cylinder_z_max,
        wind_direction=wind_direction,
        shell_offsets=shell_offsets,
        min_drone_clearance=min_drone_clearance,
        min_measurement_spacing=min_measurement_spacing,
    )

    results_df = pd.DataFrame(
        valid_points,
        columns=["x-target", "y-target", "z-target"],
    )
    if sample_groups is not None:
        results_df["sample-group"] = np.asarray(sample_groups, dtype=object)
    return results_df, bounds


# =============================================================================
# Main sample()
# =============================================================================

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
    wind_direction=(1.0, 0.0, 0.0),
    shell_offsets=(0.50, 0.75, 1.00),
    min_drone_clearance=0.50,
    min_measurement_spacing=0.50,
):
    """
    Sample CFD values at target points.

    method:
        "CSV", "array", "random", "cv", "wake_rmse", "force_cv", or
        "force_wake".

        "cv" samples points uniformly on the lateral wall of a vertical
        cylinder centred at (x=0, y=0), spanning z in [cylinder_z_min,
        cylinder_z_max]. The cylinder radius is auto-computed from the mesh
        bounding box unless `cylinder_radius` is given explicitly.

        "wake_rmse" is a velocity-field strategy: unstructured wake core,
        shear, and recovery points plus near-body and freestream anchors.

        "force_cv" is a force strategy: dense points on a near-body control
        volume, plus separation and upstream guards so the GP is constrained
        near the body without pretending the wake is fully known.

        "force_wake" is the earlier hybrid strategy kept for comparison.

    max_points:
        Hard cap on returned samples.

    prior_fn:
        Kept for API compatibility; not used by the CV sampler.
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

    point_cap = int(min(num_samples, max_points))
    rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------------
    # Target points
    # -------------------------------------------------------------------------
    sample_groups = None

    if method in ("cv", "wake_rmse", "force_cv", "force_wake"):
        valid_points, sample_groups = _named_strategy_points(
            method,
            stl_mesh,
            point_cap,
            bounds,
            epsilon,
            use_signed_distance,
            rng,
            cylinder_radius=cylinder_radius,
            cylinder_z_min=cylinder_z_min,
            cylinder_z_max=cylinder_z_max,
            wind_direction=wind_direction,
            shell_offsets=shell_offsets,
            min_drone_clearance=min_drone_clearance,
            min_measurement_spacing=min_measurement_spacing,
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
                batch,
                stl_mesh,
                epsilon=epsilon,
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
            raise ValueError(
                "method must be 'CSV', 'array', 'random', 'cv', 'wake_rmse', "
                "'force_cv', or 'force_wake'."
            )

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
                valid_points,
                stl_mesh,
                epsilon=epsilon,
                use_signed_distance=use_signed_distance,
            )
            valid_points = valid_points[mask]
        if len(valid_points) == 0:
            raise ValueError("No points left to sample.")

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
    if sample_groups is not None:
        results_df["sample-group"] = np.asarray(sample_groups, dtype=object)

    before = len(results_df)
    results_df = results_df.dropna().reset_index(drop=True)
    dropped = before - len(results_df)

    if len(results_df) > max_points:
        results_df = results_df.iloc[:max_points].reset_index(drop=True)

    print("\nSampling diagnostics")
    print("--------------------")
    print(f"CFD source points:          {len(source_coords)}")
    print(f"Candidate valid points:     {before}")
    print(f"Dropped after griddata NaN: {dropped}")
    print(f"Returned samples:           {len(results_df)} (cap {max_points})")
    print(f"Bounds:\n{bounds}")
    if "sample-group" in results_df.columns:
        print("Sample groups:")
        counts = results_df["sample-group"].value_counts()
        for group, count in counts.items():
            pct = 100.0 * count / max(len(results_df), 1)
            print(f"  {group}: {count} ({pct:.1f}%)")

    if len(results_df) == 0:
        raise RuntimeError("All sampled points became NaN after interpolation.")

    return results_df, bounds

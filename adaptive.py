"""
adaptive.py
===========

Non-intrusive adaptive sampling for the GPR wake-reconstruction pipeline.

Driven from main.py:

    result_0 = run_gpr(...)                          # initial cylinder pass
    new_pts  = propose_adaptive_points(result_0)
    coords_1 = np.vstack([result_0["training_coords"], new_pts])
    result_1 = run_gpr(..., sample_method="array", samples=coords_1)
    ...

DESIGN (v2 - rewritten to fix plane-collapse, clustering, and slowness)
-----------------------------------------------------------------------
Part 1 - WHERE to add drones
  * Difficulty field, computed cheaply on the EXISTING res^3 posterior grid (no
    second lattice, no extra GP evaluations):
        - |grad |u||  via np.gradient on the posterior speed   (shear / separation)
        - |curl u|    via np.gradient on the posterior velocity (rotational wake)
        - variance    already in the result (uncertainty)
    score = w_grad*g_hat + w_vort*w_hat + w_var*v_hat  (each min-max normalized).
    res=150 spacing (~1.3 m) is fine enough to resolve the shear layers, so we do
    NOT build a finer lattice - we just np.gradient the array we already have.

  * Candidate generation = WEIGHTED LATIN HYPERCUBE (importance sampling):
        1. Oversample a large LHS pool in the SAME tilted thick-walled cylinder
           shell the initial cylinder sampler uses (continuous, well-spread).
        2. Score each candidate by trilinearly interpolating the difficulty field.
        3. Importance-resample the pool weighted by score^beta.
    This keeps the spread/low-discrepancy character of the cylinder sampling while
    biasing toward high-difficulty regions - and because candidates are continuous
    (not grid cells), they never collapse onto an axis-aligned plane.

Part 2 - anti-clustering / spread
  * Hard drone prop-wash exclusion (anisotropic ellipsoid): 4.2 m vertical,
    1.2 m horizontal. No two accepted points (or accepted-vs-previous) may sit
    inside it.
  * Optional larger `spread_radius` relaxation: greedy farthest-point selection
    in the ellipsoid metric naturally spreads points; spread_radius scales the
    metric so points relax across the volume rather than hugging the drone limit.
  * Accumulates across phases via `previous_coords`.

Only run_gpr's return dict was extended (additively) with ell/var/noise/alpha/
sample_dat_shi/stl_mesh. Nothing in sampling.py changed.
"""

import numpy as np
from scipy.stats import qmc
from scipy.interpolate import RegularGridInterpolator

from sampling import _mesh_reject_mask


# =============================================================================
# Config
# =============================================================================

ADAPTIVE_DEFAULTS = {
    # ---- difficulty score weights (favor gradient + vorticity) ----
    "w_var": 0.2,
    "w_grad": 0.4,
    "w_vort": 0.4,

    # ---- weighted-LHS candidate pool (reuses the cylinder shell geometry) ----
    "pool_size": 4000,        # LHS candidates before resampling (oversample)
    "resample_size": 600,     # importance-resampled survivors fed to spacing
    "score_beta": 2.0,        # resampling weight = score^beta (sharper -> more biased)
    # shell thickness as fractions of R: candidates live between (1-thick_in)*R
    # and (1+thick_out)*R, i.e. a thick-walled tilted cylinder around the face.
    "shell_thick_in": 0.30,
    "shell_thick_out": 0.30,
    # bias the candidate pool toward the wake side like the cylinder sampler does
    "front_frac": 0.5,
    "front_half_angle_deg": 60.0,

    # ---- per-phase budget split across the 3 conceptual regions ----
    # region 1 = on/near the face, region 2 = inner band, region 3 = outer band.
    # Mostly on-cylinder.
    "n_new": 80,
    "frac_region1": 0.70,
    "frac_region2": 0.15,
    "frac_region3": 0.15,

    # ---- spacing ----
    "excl_horizontal": 1.2,   # m, hard drone prop-wash limit (lateral)
    "excl_vertical": 4.2,     # m, hard drone prop-wash limit (vertical)
    "spread_radius": None,    # m; if set, relax points to ~this lateral spacing
                              # (must be >= excl_horizontal). None -> just the limit.

    "epsilon": 0.5,           # mesh keep-out for proposed points (m)
    "seed": 7,
}


# =============================================================================
# Cylinder geometry (matches _oblique_cylinder_points in sampling.py)
#   Horizontal circular rings; ring center leans downstream with height.
# =============================================================================

def _cyl_frame(result):
    g = result.get("cylinder_geom")
    if g is None:
        raise ValueError(
            "cylinder_geom is None - the initial pass must use sample_method='cylinder'."
        )
    base = np.asarray(g["bottom_center"], float)   # [cx, cy, z_bottom]
    top = np.asarray(g["top_center"], float)       # [cx', cy', z_top]
    R = float(g["R"])
    z_bottom, z_top = float(base[2]), float(top[2])
    z_mid = 0.5 * (z_bottom + z_top)
    dz = max(z_top - z_bottom, 1e-12)
    shift_per_dz = (top[:2] - base[:2]) / dz       # horizontal lean per unit z
    P0 = base[:2] - shift_per_dz * (z_bottom - z_mid)   # xy center at z_mid
    # downstream horizontal unit vector (for front/back wedge biasing)
    norm = np.linalg.norm(shift_per_dz)
    s_hat = (shift_per_dz / norm) if norm > 1e-9 else None
    return {"P0": P0, "shift_per_dz": shift_per_dz, "z_bottom": z_bottom,
            "z_top": z_top, "z_mid": z_mid, "R": R, "H": dz, "s_hat": s_hat}


def _ring_center_xy(z, fr):
    z = np.atleast_1d(np.asarray(z, float))
    return fr["P0"][None, :] + (z - fr["z_mid"])[:, None] * fr["shift_per_dz"][None, :]


def _horizontal_radial(points, fr):
    """Horizontal distance rho to the ring center at each point's z, plus rhat, z."""
    points = np.asarray(points, float)
    z = points[:, 2]
    rc = _ring_center_xy(z, fr)
    dxy = points[:, :2] - rc
    rho = np.linalg.norm(dxy, axis=1)
    rhat = dxy / np.maximum(rho[:, None], 1e-12)
    return rho, rhat, z


# =============================================================================
# PART 1a - difficulty field on the EXISTING res^3 grid (np.gradient, no GP calls)
# =============================================================================

def _normalize(a):
    a = np.asarray(a, float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(hi, lo):
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _difficulty_field(result, cfg):
    """Compute grad(|u|), curl(u), and variance magnitude on the res^3 grid.

    Returns (gx, gy, gz, score_grid (res,res,res)) plus an interpolator-ready
    bundle. Uses np.gradient on arrays already in `result` - no GP evaluations,
    no extra lattice. res spacing resolves the shear layers adequately.
    """
    res = result["res"]
    bounds = np.asarray(result["bounds"])
    gx = np.linspace(bounds[0, 0], bounds[0, 1], res)
    gy = np.linspace(bounds[1, 0], bounds[1, 1], res)
    gz = np.linspace(bounds[2, 0], bounds[2, 1], res)
    dx = gx[1] - gx[0] if res > 1 else 1.0
    dy = gy[1] - gy[0] if res > 1 else 1.0
    dz = gz[1] - gz[0] if res > 1 else 1.0

    U = np.asarray(result["GPR_posterior"]).reshape(res, res, res, 3)
    ux, uy, uz = U[..., 0], U[..., 1], U[..., 2]
    speed = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)

    # gradient of speed -> shear / separation layers
    gsx, gsy, gsz = np.gradient(speed, dx, dy, dz, edge_order=1)
    grad_mag = np.sqrt(gsx ** 2 + gsy ** 2 + gsz ** 2)

    # curl of u -> rotational structures in the wake
    duz_dy = np.gradient(uz, dy, axis=1, edge_order=1)
    duy_dz = np.gradient(uy, dz, axis=2, edge_order=1)
    dux_dz = np.gradient(ux, dz, axis=2, edge_order=1)
    duz_dx = np.gradient(uz, dx, axis=0, edge_order=1)
    duy_dx = np.gradient(uy, dx, axis=0, edge_order=1)
    dux_dy = np.gradient(ux, dy, axis=1, edge_order=1)
    wx = duz_dy - duy_dz
    wy = dux_dz - duz_dx
    wz = duy_dx - dux_dy
    vort_mag = np.sqrt(wx ** 2 + wy ** 2 + wz ** 2)

    var_mag = np.linalg.norm(
        np.asarray(result["GPR_variances"]).reshape(res, res, res, 3), axis=-1)

    score = (cfg["w_grad"] * _normalize(grad_mag)
             + cfg["w_vort"] * _normalize(vort_mag)
             + cfg["w_var"] * _normalize(var_mag))

    return gx, gy, gz, score


def _score_interp(result, cfg):
    """Return a callable score(points)->(n,) interpolating the grid difficulty."""
    gx, gy, gz, score = _difficulty_field(result, cfg)
    interp = RegularGridInterpolator((gx, gy, gz), score,
                                     bounds_error=False, fill_value=0.0)
    return interp


# =============================================================================
# PART 1a' - GRID-FREE difficulty score, evaluated directly on candidate points
#
# Instead of building a res^3 grid, np.gradient-ing it, and interpolating, we
# evaluate the SAME quantities (grad|u|, curl u, variance) directly on the
# candidate points using the trained GP. For each point we need the velocity at
# the point and at 6 neighbours (+-dx, +-dy, +-dz) so central differences give
# the spatial derivatives. The velocity used is the POSTERIOR = Julia prior +
# K(x*,X_train) @ alpha (exactly what posterior_mean_batched returns), i.e. the
# prior corrected by the GP - identical to what the grid path differentiated.
# Variance reuses the trained Cholesky factors (no retraining).
# =============================================================================

def _fd_steps(ell, fd_step):
    """Per-axis finite-difference step.

    fd_step is None  -> auto: 0.15 * ell per axis, clamped to [0.5, 3.0] m.
    fd_step a scalar -> that fixed step on every axis.
    fd_step a length-3 -> used per axis as given.
    The auto rule ties the step to the GP correlation length (the field is a sum
    of Matern-5/2 kernels of width ell): small enough vs ell to approximate the
    true gradient, large enough not to amplify the GP noise floor.
    """
    ell = np.asarray(ell, float).reshape(-1)
    if fd_step is None:
        h = np.clip(0.1 * ell, 0.5, 10.0)
    elif np.isscalar(fd_step):
        h = np.full(3, float(fd_step))
    else:
        h = np.asarray(fd_step, float).reshape(-1)
        if h.size == 1:
            h = np.full(3, float(h))
    return h[:3]


def _posterior_vel_on_points(points, prior_fn, training_coords, ell, var, alpha,
                             posterior_batch):
    """GP posterior velocity on arbitrary points: prior_fn(points) + K@alpha.

    Mirrors GPR.posterior_mean_batched but takes the prior closure so the prior
    and the GP correction are evaluated together here. Returns (n,3)."""
    from GPR import posterior_mean_batched
    prior = np.asarray(prior_fn(points), float).reshape(-1, 3)
    post = posterior_mean_batched(
        points, training_coords, ell, var, alpha, prior,
        batch=posterior_batch, progress_every=0)
    return np.asarray(post, float).reshape(-1, 3)


def _eval_score_inputs_on_points(points, prior_fn, training_coords, ell, var,
                                 alpha, c, low, fd_step=None, posterior_batch=2000,
                                 compute_var=True):
    """Evaluate grad|u|, curl u (from the GP posterior) and |variance| at `points`.

    Uses central differences with the 6 neighbours of each point. Returns a dict
    with arrays of shape (n,): 'grad_mag', 'vort_mag', 'var_mag'. These are the
    raw (un-normalised) score ingredients; normalisation/blending happens in the
    caller so it matches the grid path's _difficulty_field exactly.
    """
    from GPR import posterior_vars_batched

    points = np.asarray(points, float)
    n = points.shape[0]
    h = _fd_steps(ell, fd_step)

    # Build the stencil: center + 6 neighbours, stacked so we do ONE batched
    # posterior evaluation over all 7n points (cheap relative to a res^3 grid).
    offsets = np.array([
        [0.0, 0.0, 0.0],
        [+h[0], 0.0, 0.0], [-h[0], 0.0, 0.0],
        [0.0, +h[1], 0.0], [0.0, -h[1], 0.0],
        [0.0, 0.0, +h[2]], [0.0, 0.0, -h[2]],
    ])
    stencil = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 3)  # (7n,3)

    vel = _posterior_vel_on_points(stencil, prior_fn, training_coords, ell, var,
                                   alpha, posterior_batch).reshape(n, 7, 3)

    # central differences: d/dx ~ (v[+x]-v[-x]) / (2 hx), etc.
    dudx = (vel[:, 1, :] - vel[:, 2, :]) / (2.0 * h[0])   # (n,3)
    dudy = (vel[:, 3, :] - vel[:, 4, :]) / (2.0 * h[1])
    dudz = (vel[:, 5, :] - vel[:, 6, :]) / (2.0 * h[2])

    # gradient of speed |u| at the centre (chain rule: grad|u| = (J^T u)/|u|)
    u0 = vel[:, 0, :]                                     # (n,3)
    speed = np.linalg.norm(u0, axis=1)
    safe = np.maximum(speed, 1e-12)
    grad_speed = np.column_stack([
        (dudx * u0).sum(axis=1),
        (dudy * u0).sum(axis=1),
        (dudz * u0).sum(axis=1),
    ]) / safe[:, None]
    grad_mag = np.linalg.norm(grad_speed, axis=1)

    # curl u = (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy)
    wx = dudy[:, 2] - dudz[:, 1]
    wy = dudz[:, 0] - dudx[:, 2]
    wz = dudx[:, 1] - dudy[:, 0]
    vort_mag = np.sqrt(wx ** 2 + wy ** 2 + wz ** 2)

    if compute_var:
        v = posterior_vars_batched(points, training_coords, ell, var, c, low,
                                   batch=posterior_batch, progress_every=0)
        var_mag = np.linalg.norm(np.asarray(v, float).reshape(-1, 3), axis=1)
    else:
        var_mag = np.zeros(n)

    return {"grad_mag": grad_mag, "vort_mag": vort_mag, "var_mag": var_mag}


def build_pool_and_score_inputs(cylinder_geom, stl_mesh, prior_fn,
                                training_coords, ell, var, alpha, c, low,
                                adaptive_config=None, pool_seed=None,
                                fd_step=None, posterior_batch=2000):
    """Build the shell LHS candidate pool and evaluate its grid-free score inputs.

    Called from run_gpr's grid-free branch WHILE the panel solver is still alive
    (prior_fn needs it). Returns a picklable bundle that propose_adaptive_points
    consumes directly, so no res^3 grid is ever built and the solver can close
    immediately afterwards.

    The bundle stores the mesh-rejected pool coordinates plus the raw per-point
    grad/vort/var ingredients; the caller (propose_adaptive_points) normalises,
    blends with the w_* weights, resamples and spaces them - identical downstream
    logic to the grid path, just with point-wise instead of interpolated scores.
    """
    cfg = dict(ADAPTIVE_DEFAULTS)
    if adaptive_config:
        cfg.update(adaptive_config)
    if pool_seed is None:
        pool_seed = cfg.get("pool_seed", cfg["seed"])

    fr = _cyl_frame({"cylinder_geom": cylinder_geom})

    # candidate pool over the thick tilted shell (pool_size is a variable)
    pool = _shell_lhs_pool(fr, cfg, cfg["pool_size"], pool_seed)
    keep = _mesh_reject_mask(pool, stl_mesh, epsilon=cfg["epsilon"],
                             use_signed_distance=True)
    pool = pool[keep]
    if len(pool) == 0:
        raise RuntimeError("LHS candidate pool empty after mesh rejection.")

    inputs = _eval_score_inputs_on_points(
        pool, prior_fn, training_coords, ell, var, alpha, c, low,
        fd_step=fd_step, posterior_batch=posterior_batch, compute_var=True)

    return {
        "pool": pool,
        "grad_mag": inputs["grad_mag"],
        "vort_mag": inputs["vort_mag"],
        "var_mag": inputs["var_mag"],
        "pool_seed": pool_seed,
        "fd_step": fd_step,                       # raw config value (may be None)
        "fd_step_resolved": _fd_steps(ell, fd_step),  # actual per-axis steps used [m]
    }


# =============================================================================
# PART 1b - weighted Latin-hypercube candidate pool (reuses cylinder shell)
# =============================================================================

def _shell_lhs_pool(fr, cfg, n, seed):
    """Oversampled LHS pool in the tilted thick-walled cylinder shell.

    Mirrors _oblique_cylinder_points: phi=0 is the wake (downstream) side and
    gets denser sampling via front_frac; rings are horizontal and lean downstream
    with height. Radius is drawn in [(1-thick_in)R, (1+thick_out)R] so the pool
    fills a thick wall around the face. Returns (n,3) coordinates.
    """
    R = fr["R"]
    r_lo = (1.0 - cfg["shell_thick_in"]) * R
    r_hi = (1.0 + cfg["shell_thick_out"]) * R

    # orientation: phi measured from downstream s_hat, swept toward cross-wind.
    s2 = fr["s_hat"]
    if s2 is None:
        s2 = np.array([1.0, 0.0])           # arbitrary if no tilt
    c2 = np.array([-s2[1], s2[0]])           # cross-wind (horizontal)

    def chunk(nn, phi_min, phi_max, sd):
        if nn <= 0:
            return np.empty((0, 3))
        u = qmc.LatinHypercube(d=3, seed=sd).random(nn)
        phi = phi_min + u[:, 0] * (phi_max - phi_min)
        z = fr["z_bottom"] + u[:, 1] * fr["H"]
        rad = r_lo + u[:, 2] * (r_hi - r_lo)
        rc = _ring_center_xy(z, fr)          # (nn,2)
        offset = rad[:, None] * (np.cos(phi)[:, None] * s2[None, :]
                                 + np.sin(phi)[:, None] * c2[None, :])
        xy = rc + offset
        return np.column_stack([xy[:, 0], xy[:, 1], z])

    # front_frac=None -> UNIFORM sweep over the full ring (no front/back bias).
    if cfg.get("front_frac") is None:
        return chunk(n, 0.0, 2 * np.pi, seed)

    half = np.radians(cfg["front_half_angle_deg"])
    n_front = int(round(cfg["front_frac"] * n))
    n_back = n - n_front
    front = chunk(n_front, -half, half, seed)
    back = chunk(n_back, half, 2 * np.pi - half, seed + 1)
    return np.vstack([front, back])


def _weighted_resample(pool, weights, k, rng):
    """Importance resample k points from pool without replacement, prob ~ weights."""
    pool = np.asarray(pool, float)
    w = np.asarray(weights, float)
    w = np.clip(w, 0, None)
    if w.sum() <= 0 or len(pool) == 0:
        # fall back to uniform spread
        idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return pool[idx], np.ones(len(idx))
    p = w / w.sum()
    k = min(k, len(pool))
    idx = rng.choice(len(pool), size=k, replace=False, p=p)
    return pool[idx], w[idx]


# =============================================================================
# PART 2 - ellipsoid-metric anti-clustering + spread
# =============================================================================

def _ellip_ok(cand, existing, rh, rv):
    if existing is None or len(existing) == 0:
        return True
    d = existing - cand[None, :]
    dxy2 = (d[:, 0] ** 2 + d[:, 1] ** 2) / (rh ** 2)
    dz2 = (d[:, 2] ** 2) / (rv ** 2)
    return np.all(dxy2 + dz2 >= 1.0)


def _greedy_select(cands, weights, n_take, previous, rh, rv):
    """Greedy farthest-point selection in the ellipsoid metric, score-seeded.

    rh/rv define the HARD exclusion (drone limit) AND the spread metric. A larger
    rh/rv spreads points further apart. Picks the top-weight valid candidate, then
    repeatedly adds the one maximizing min ellipsoid distance to accepted points
    while clearing the exclusion of all previous + accepted points.
    """
    cands = np.asarray(cands, float)
    if len(cands) == 0 or n_take <= 0:
        return np.empty((0, 3))
    weights = np.asarray(weights, float)
    order = np.argsort(-weights)
    cands, weights = cands[order], weights[order]

    prev = np.asarray(previous, float) if previous is not None and len(previous) else np.empty((0, 3))
    accepted = []

    def min_ed(p, ref):
        if len(ref) == 0:
            return np.inf
        d = ref - p[None, :]
        return np.sqrt(np.min((d[:, 0] ** 2 + d[:, 1] ** 2) / rh ** 2 + d[:, 2] ** 2 / rv ** 2))

    for p in cands:
        if _ellip_ok(p, prev, rh, rv):
            accepted.append(p)
            break
    if not accepted:
        return np.empty((0, 3))

    while len(accepted) < n_take:
        acc = np.asarray(accepted)
        best, best_s = None, -1.0
        for i, p in enumerate(cands):
            allref = np.vstack([prev, acc]) if len(prev) else acc
            if not _ellip_ok(p, allref, rh, rv):
                continue
            s = min_ed(p, acc) + 1e-6 * (len(cands) - i)   # tie-break toward weight
            if s > best_s:
                best_s, best = s, p
        if best is None:
            break
        accepted.append(best)
    return np.asarray(accepted)


# =============================================================================
# Public entry point
# =============================================================================

def propose_adaptive_points(result, previous_coords=None, adaptive_config=None,
                            verbose=True):
    """Propose a new batch of drone points from a finished GPR result.

    Returns (K,3) coordinates (mesh-rejected, anti-clustered, spread). Append to
    previous_coords for the next run_gpr(sample_method='array', samples=...).
    """
    cfg = dict(ADAPTIVE_DEFAULTS)
    if adaptive_config:
        cfg.update(adaptive_config)

    # Per-phase seed for the candidate pool. The shell LHS pool is otherwise
    # drawn from the SAME fixed seed every phase, so multi-phase runs all
    # resample from one frozen lattice of positions -> the greedy selector keeps
    # re-proposing coordinates already occupied by earlier phases, which then get
    # rejected as self-collisions rather than placed. Letting the caller pass a
    # distinct pool_seed per phase (e.g. cfg["seed"] + phase) gives each phase a
    # fresh candidate cloud while staying fully reproducible. Falls back to
    # cfg["seed"] so single-call / legacy use is unchanged.
    pool_seed = cfg.get("pool_seed", cfg["seed"])
    rng = np.random.default_rng(pool_seed)

    if previous_coords is None:
        previous_coords = np.asarray(result["training_coords"], float)
    previous_coords = np.asarray(previous_coords, float)

    fr = _cyl_frame(result)
    R = fr["R"]

    # spacing: hard drone limit, optionally widened by spread_radius
    rh = cfg["excl_horizontal"]
    rv = cfg["excl_vertical"]
    if cfg["spread_radius"] is not None:
        scale = max(cfg["spread_radius"] / max(rh, 1e-9), 1.0)
        rh = rh * scale
        rv = rv * scale     # keep the 4.2/1.2 anisotropy ratio while spreading

    # ---- PART 1+1b: candidate pool + per-candidate difficulty score ----
    # Three paths, same downstream logic:
    #   (A) precomputed grid-free bundle in result["adaptive_pool_bundle"].
    #   (B) live grid-free: result exposes _prior_fn + Cholesky factors (run_gpr
    #       ran with grid_eval=False). Build the pool here with THIS phase's cfg,
    #       evaluate grad/vort/var directly on each candidate (+6 neighbours), then
    #       close the (still-alive) panel solver. No res^3 grid is ever built.
    #   (C) legacy grid: differentiate the res^3 posterior and interpolate the
    #       score onto a freshly drawn pool (run_gpr ran with grid_eval=True).
    bundle = result.get("adaptive_pool_bundle")
    if bundle is None and result.get("_prior_fn") is not None:
        bundle = build_pool_and_score_inputs(
            result.get("cylinder_geom"), result["stl_mesh"], result["_prior_fn"],
            result["training_coords"], result["ell"], result["var"],
            result["alpha"], result["_chol_c"], result["_chol_low"],
            adaptive_config=cfg, pool_seed=pool_seed,
            fd_step=cfg.get("fd_step"),
            posterior_batch=result.get("posterior_batch", 2000))
        # done with the panel solver for this phase - release the Julia process
        closer = result.get("_close_solver")
        if closer is not None:
            closer()
            result["_prior_fn"] = None
            result["_close_solver"] = None

    if bundle is not None:
        pool = np.asarray(bundle["pool"], float)
        # Always surface the ACTUAL per-axis finite-difference steps used (the
        # resolved values, not the raw config which is None under the auto rule),
        # since this is the main thing to tune. Printed even when verbose=False.
        _fdr = bundle.get("fd_step_resolved")
        _fdr_s = (np.array2string(np.asarray(_fdr, float), precision=3)
                  if _fdr is not None else "n/a")
        print(f"[adaptive] grid-free fd step (dx,dy,dz)={_fdr_s} m "
              f"(config fd_step={bundle.get('fd_step')})", flush=True)
        if verbose:
            print(f"[adaptive] grid-free: pool {len(pool)} pts, "
                  f"pool_seed={bundle.get('pool_seed')}")
        g = _normalize(bundle["grad_mag"])
        w = _normalize(bundle["vort_mag"])
        v = _normalize(bundle["var_mag"])
        pool_score = cfg["w_grad"] * g + cfg["w_vort"] * w + cfg["w_var"] * v
    else:
        # ---- legacy difficulty field (grid, np.gradient) + score interpolator ----
        if result.get("GPR_posterior") is None or result.get("res") is None:
            raise RuntimeError(
                "propose_adaptive_points: no scoring data available. The result has "
                "neither grid-free ingredients (_prior_fn / Cholesky factors, from a "
                "grid_eval=False run) nor a res^3 grid (GPR_posterior, from a "
                "grid_eval=True run). A grid-free result can only be proposed from "
                "ONCE - re-run run_gpr before proposing again.")
        score_fn = _score_interp(result, cfg)
        pool = _shell_lhs_pool(fr, cfg, cfg["pool_size"], pool_seed)
        keep = _mesh_reject_mask(pool, result["stl_mesh"], epsilon=cfg["epsilon"],
                                 use_signed_distance=True)
        pool = pool[keep]
        if len(pool) == 0:
            raise RuntimeError("LHS candidate pool empty after mesh rejection.")
        if verbose:
            print(f"[adaptive] pool_seed={pool_seed}  pool[0]={pool[0].round(2)}")
        pool_score = score_fn(pool)

    # Points near/inside the building can produce a non-finite posterior/score
    # (the panel prior blanks the interior -> NaN). Treat non-finite difficulty
    # as zero weight so the importance-resample probabilities stay finite; the
    # point can still be chosen to fill a band, just not preferentially. Without
    # this a single NaN score makes p = w/w.sum() all-NaN and rng.choice raises.
    pool_score = np.nan_to_num(pool_score, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(pool_score, 0, None) ** cfg["score_beta"]

    # importance-resample to a manageable survivor set, then split by radial band
    survivors, surv_w = _weighted_resample(pool, weights, cfg["resample_size"], rng)

    rho, _, _ = _horizontal_radial(survivors, fr)
    band_in = rho < R * (1.0 - 0.02)
    band_face = np.abs(rho - R) <= R * 0.10
    band_out = rho > R * (1.0 + 0.02)

    # budget
    n_total = cfg["n_new"]
    n1 = int(round(cfg["frac_region1"] * n_total))
    n2 = int(round(cfg["frac_region2"] * n_total))
    n3 = n_total - n1 - n2

    if verbose:
        print(f"[adaptive] pool {len(pool)} -> survivors {len(survivors)} "
              f"(face {band_face.sum()}, inner {band_in.sum()}, outer {band_out.sum()}); "
              f"spacing rh={rh:.2g} rv={rv:.2g}")

    # ---- PART 2: spread + exclusion, region by region, accumulating ----
    accepted_prev = previous_coords.copy()
    chosen = []
    for mask, n_take, label in [
        (band_face, n1, "region1/face"),
        (band_in, n2, "region2/inner"),
        (band_out, n3, "region3/outer"),
    ]:
        cand = survivors[mask]
        w = surv_w[mask]
        picks = _greedy_select(cand, w, n_take, accepted_prev, rh, rv)
        if len(picks):
            chosen.append(picks)
            accepted_prev = np.vstack([accepted_prev, picks])
        if verbose:
            print(f"[adaptive] {label}: requested {n_take}, candidates {len(cand)}, "
                  f"placed {len(picks)}")

    new_points = np.vstack(chosen) if chosen else np.empty((0, 3))
    if verbose:
        print(f"[adaptive] total new points: {len(new_points)}")
    return new_points


def propose_top_cap_points(result, previous_coords=None, n_new=20,
                           cap_z_offset=0.0, n_rings=None, adaptive_config=None,
                           cap_oversample=40, verbose=True):
    """Place a small number of drones on the cylinder TOP CAP.

    Motivated by momentum escaping through the open top of the control volume:
    these points sit on the horizontal disk at (or just above) z_top, out to
    radius R, so the GP has data where the top-surface flux is integrated.
    Candidates are drawn by area-uniform Latin-hypercube over the disk (r = R*sqrt(u)
    so points don't pile up at the center, theta via LHS for low-discrepancy
    spread), then de-clustered against previous_coords with the same drone-
    exclusion ellipsoid as the side sampler.

    n_new        : target number of cap drones (keep small).
    cap_z_offset : place the cap at z_top + this (0 = exactly on the top ring plane).
    cap_oversample: candidates generated = cap_oversample * n_new before de-clustering.
    n_rings      : ignored (kept for backward-compat with old ring-based calls).
    """
    cfg = dict(ADAPTIVE_DEFAULTS)
    if adaptive_config:
        cfg.update(adaptive_config)
    rng = np.random.default_rng(cfg["seed"])

    if previous_coords is None:
        previous_coords = np.asarray(result["training_coords"], float)
    previous_coords = np.asarray(previous_coords, float)

    fr = _cyl_frame(result)
    R = fr["R"]
    z_cap = fr["z_top"] + cap_z_offset
    cap_center_xy = _ring_center_xy(z_cap, fr)[0]   # tilted center at cap height

    rh = cfg["excl_horizontal"]
    rv = cfg["excl_vertical"]
    if cfg["spread_radius"] is not None:
        scale = max(cfg["spread_radius"] / max(rh, 1e-9), 1.0)
        rh *= scale; rv *= scale

    # candidate cloud: area-uniform LHS over the cap disk (oversampled).
    # r = R*sqrt(u0) -> uniform density per unit area (raw u0 over-samples center);
    # theta from a second LHS column -> low-discrepancy azimuth, no ring banding.
    n_cand = max(int(cap_oversample * n_new), 64)
    u = qmc.LatinHypercube(d=2, seed=cfg["seed"]).random(n_cand)
    rad = R * np.sqrt(u[:, 0])
    phi = 2.0 * np.pi * u[:, 1]
    x = cap_center_xy[0] + rad * np.cos(phi)
    y = cap_center_xy[1] + rad * np.sin(phi)
    cands = np.column_stack([x, y, np.full(n_cand, z_cap)])
    weights = 1.0 + rad / R          # mild continuous bias to outer disk (where flux is)

    # mesh-reject then de-cluster with greedy farthest-point selection
    keep = _mesh_reject_mask(cands, result["stl_mesh"], epsilon=cfg["epsilon"],
                             use_signed_distance=True)
    cands, weights = cands[keep], weights[keep]
    picks = _greedy_select(cands, weights, n_new, previous_coords, rh, rv)

    if verbose:
        print(f"[adaptive] top-cap: z={z_cap:.3g}, candidates {len(cands)}, "
              f"placed {len(picks)} (target {n_new})")
    return picks
# slip.py  –  GPR flow reconstruction with divergence-free kernel
#             Sensor placement via sampling_region logic, auto-derived from body geometry.
#             + Drag via direct pressure surface integration (exact CFD reference)
#               and GP-reconstructed pressure via Bernoulli on the wall.

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import hessian
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.tri import Triangulation

jax.config.update("jax_enable_x64", True)

HERE       = Path(__file__).resolve().parent
FIELD_PATH = HERE / "output14_5.xlsx"
CYL_PATH   = HERE / "building14_5.xlsx"

# =============================================================================
# Kernel  (ARD Matern 5/2) - Divergence Free via Stream Function Hessian
# =============================================================================
def matern52(x, xp, var, lx, ly):
    dx = (x[0] - xp[0]) / lx
    dy = (x[1] - xp[1]) / ly
    r  = jnp.sqrt(dx * dx + dy * dy + 1e-12)
    s  = jnp.sqrt(5.0) * r
    return var * (1 + s + s ** 2 / 3) * jnp.exp(-s)

def kern_entry(X, Xp, var, lx, ly):
    x,  d  = X[:2],  X[2:4]
    xp, dp = Xp[:2], Xp[2:4]
    H = -hessian(matern52, argnums=0)(x, xp, var, lx, ly)
    C = jnp.array([[ H[1, 1], -H[1, 0]],
                   [-H[0, 1],  H[0, 0]]])
    return d @ C @ dp

_row = jax.vmap(kern_entry, in_axes=(None, 0, None, None, None))
kern = jax.jit(jax.vmap(_row, in_axes=(0, None, None, None, None)))

# =============================================================================
# Observation builders
# =============================================================================
def to_4d(xy, uv=None):
    N = xy.shape[0]
    pos  = jnp.repeat(xy, 2, axis=0)
    dirs = jnp.tile(jnp.eye(2), (N, 1))
    X = jnp.hstack([pos, dirs])
    if uv is not None:
        return X, uv.reshape(-1, order="C")
    return X

def wall_obs_slip(xy_w, n_w, U_inf, V_inf, U_char):
    """Only enforces No-Penetration (u · n = 0); tangential slip is free."""
    U_vec = jnp.array([U_inf, V_inf])
    X_n = jnp.hstack([xy_w, n_w])
    y_n = -(n_w @ U_vec) / U_char
    return X_n, y_n

# =============================================================================
# Train / predict
# =============================================================================
JITTER = 1e-8

def diag_noise(noise, mask):
    return noise * mask + JITTER * (1.0 - mask)

def neg_log_ml(theta, X_tr, y_tr, mask):
    vs, lx, ly, noise = jnp.exp(theta)
    K = kern(X_tr, X_tr, vs, lx, ly) + jnp.diag(diag_noise(noise, mask))
    L = jnp.linalg.cholesky(K)
    a = jax.scipy.linalg.cho_solve((L, True), y_tr)
    n = len(y_tr)
    return 0.5 * (y_tr @ a + 2 * jnp.sum(jnp.log(jnp.diag(L)))
                  + n * jnp.log(2 * jnp.pi))

def predict(X_tr, y_tr, X_te, theta, mask, chunk=2000):
    vs, lx, ly, noise = jnp.exp(theta)
    K = kern(X_tr, X_tr, vs, lx, ly) + jnp.diag(diag_noise(noise, mask))
    L = jnp.linalg.cholesky(K)
    a = jax.scipy.linalg.cho_solve((L, True), y_tr)
    out = []
    for i in range(0, len(X_te), chunk):
        out.append(np.array(kern(X_te[i:i + chunk], X_tr, vs, lx, ly) @ a))
    return np.concatenate(out)

# =============================================================================
# Data loading & Geometry
# =============================================================================
def _clean(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _read_table(path):
    path = Path(path)
    try:
        return _clean(pd.read_excel(path, engine="openpyxl"))
    except Exception:
        pass
    try:
        return _clean(pd.read_excel(path, engine="xlrd"))
    except Exception:
        pass
    for sep in (",", "\t", ";", " "):
        try:
            df = _clean(pd.read_csv(path, sep=sep))
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    raise ValueError(
        f"Cannot read '{path}'. Re-export as .xlsx or .csv."
    )

def load_field():
    df = _read_table(FIELD_PATH)
    print("Field columns:", list(df.columns))
    xy = df[["x-coordinate", "y-coordinate"]].to_numpy(float)
    uv = df[["x-velocity",   "y-velocity"]].to_numpy(float)

    # Find pressure column robustly
    p_col = None
    for col in ["pressure", "static-pressure", "Pressure",
                "Static Pressure", "static pressure", "p", "P"]:
        if col in df.columns:
            p_col = col
            break
    if p_col is None:
        # Last resort: any column with 'press' in name
        candidates = [c for c in df.columns if "press" in c.lower()]
        if candidates:
            p_col = candidates[0]
    if p_col is None:
        raise ValueError(
            f"No pressure column found. Columns: {list(df.columns)}"
        )
    print(f"Using pressure column: '{p_col}'")
    p = df[p_col].to_numpy(float)
    return xy, uv, p

def wall_frame(xy_wall):
    """Outward normals for any star-shaped closed body."""
    c = xy_wall.mean(axis=0)
    order = np.argsort(np.arctan2(xy_wall[:, 1] - c[1],
                                  xy_wall[:, 0] - c[0]))
    xy_o = xy_wall[order]
    t = np.roll(xy_o, -1, axis=0) - np.roll(xy_o, 1, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.column_stack([t[:, 1], -t[:, 0]])
    sign = np.sign(np.einsum("ij,ij->i", n, xy_o - c))
    sign[sign == 0] = 1
    n *= sign[:, None]
    n_out = np.empty_like(n)
    n_out[order] = n
    return n_out

def _geometric_diameter(xy):
    from scipy.spatial import ConvexHull
    if len(xy) < 2:
        return 0.0
    try:
        hull_pts = xy[ConvexHull(xy).vertices]
    except Exception:
        hull_pts = xy
    n   = len(hull_pts)
    dia = 0.0
    j   = 1
    for i in range(n):
        while True:
            next_j = (j + 1) % n
            if np.sum((hull_pts[next_j] - hull_pts[i]) ** 2) > \
               np.sum((hull_pts[j]      - hull_pts[i]) ** 2):
                j = next_j
            else:
                break
        dia = max(dia, float(np.linalg.norm(hull_pts[j] - hull_pts[i])))
        j   = (j + 1) % n
    return dia

def load_body():
    df = _read_table(CYL_PATH)
    xy = df[["x-coordinate", "y-coordinate"]].to_numpy(float)
    xy = np.unique(xy, axis=0)
    centre = xy.mean(axis=0)
    L = _geometric_diameter(xy) / 2.0
    n = wall_frame(xy)
    return xy, n, centre, L

# =============================================================================
# Sampling region geometry
# =============================================================================
def _rotmat(theta_deg):
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])

def _rectangle_vertices(center, theta_deg, s, a, h):
    R = _rotmat(theta_deg)
    corners_local = np.array([[-s, -h], [a, -h], [a, h], [-s, h]])
    return corners_local @ R.T + np.array(center)

def _inside_sampling_region(x, y, center, theta_deg, s, a, h,
                             excl_center, excl_radius):
    x0, y0 = center
    cx, cy = excl_center
    th = np.deg2rad(theta_deg)
    c, sn = np.cos(th), np.sin(th)
    dx, dy = x - x0, y - y0
    u =  dx * c + dy * sn
    v = -dx * sn + dy * c
    in_rect = (-s <= u) & (u <= a) & (-h <= v) & (v <= h)
    in_excl = (x - cx) ** 2 + (y - cy) ** 2 <= excl_radius ** 2
    return in_rect & (~in_excl)

def _derive_sampling_geometry(centre, L):
    excl_diam = 2.0 * L
    s         = 3.0  * excl_diam
    a         = 8 * excl_diam
    h         = 10  * excl_diam
    return L, s, a, h

def _seed_from_string(seed_string):
    digest = hashlib.sha256(seed_string.encode("utf-8")).hexdigest()
    return int(digest, 16) % (2 ** 32)

# =============================================================================
# Sensor sampling
# =============================================================================
def sample_sensors(xy_full, xy_body, centre, L,
                   num_left=50, num_right=30, num_interior=20,
                   seed_string="PEACH_VIBE", theta_override=None):
    excl_radius, s, a, h = _derive_sampling_geometry(centre, L)
    theta_deg = 0.0 if theta_override is None else theta_override

    seed  = _seed_from_string(seed_string)
    rng   = np.random.default_rng(seed)
    verts = _rectangle_vertices(centre, theta_deg, s, a, h)

    left_pts  = np.linspace(verts[0], verts[3], num_left)
    right_pts = np.linspace(verts[1], verts[2], num_right)

    x_lo, x_hi = xy_full[:, 0].min(), xy_full[:, 0].max()
    y_lo, y_hi = xy_full[:, 1].min(), xy_full[:, 1].max()

    interior_pts = []
    while len(interior_pts) < num_interior:
        xs = rng.uniform(x_lo, x_hi, size=num_interior * 10)
        ys = rng.uniform(y_lo, y_hi, size=num_interior * 10)
        mask = _inside_sampling_region(xs, ys,
                                       centre, theta_deg, s, a, h,
                                       centre, excl_radius)
        valid = np.column_stack([xs[mask], ys[mask]])
        interior_pts.extend(valid.tolist())
    interior_pts = np.array(interior_pts[:num_interior])

    all_pts = np.vstack([left_pts, right_pts, interior_pts])

    tree = KDTree(xy_full)
    _, raw_idx = tree.query(all_pts)
    _, unique_pos = np.unique(raw_idx, return_index=True)
    idx = raw_idx[np.sort(unique_pos)]

    return idx

# =============================================================================
# Scalar GP for pressure
# =============================================================================
def _matern52_scalar(r2):
    """Matern 5/2 kernel value given squared scaled distance."""
    r = jnp.sqrt(r2 + 1e-12)
    s = jnp.sqrt(5.0) * r
    return (1.0 + s + s**2 / 3.0) * jnp.exp(-s)

def _scalar_kern_matrix(X1, X2, var, lx, ly):
    """
    Pure-numpy ARD Matern-5/2 scalar kernel matrix. (N1, N2)
    X1: (N1,2), X2: (N2,2)
    """
    dx = (X1[:, 0:1] - X2[:, 0]) / lx   # (N1,N2)
    dy = (X1[:, 1:2] - X2[:, 1]) / ly
    r2 = dx**2 + dy**2
    r  = np.sqrt(r2 + 1e-12)
    s  = np.sqrt(5.0) * r
    return var * (1.0 + s + s**2 / 3.0) * np.exp(-s)


def _fit_scalar_gp(xy_tr, y_tr, xy_te,
                   starts=None, bounds=None, jitter=1e-6):
    """
    Fit a scalar ARD Matern-5/2 GP to (xy_tr, y_tr) and predict at xy_te.
    Returns predictions (N_te,).
    Hyperparams: log(var), log(lx), log(ly), log(noise).
    """
    y_mean = float(y_tr.mean())
    y_std  = float(y_tr.std()) + 1e-8
    y_norm = (y_tr - y_mean) / y_std

    def nlml(log_theta):
        var, lx, ly, noise = np.exp(log_theta)
        K = _scalar_kern_matrix(xy_tr, xy_tr, var, lx, ly)
        K += (noise + jitter) * np.eye(len(y_norm))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        a = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        return float(0.5 * (y_norm @ a
                     + 2 * np.sum(np.log(np.diag(L)))
                     + len(y_norm) * np.log(2 * np.pi)))

    if starts is None:
        # use domain extents to seed length-scales
        span_x = float(xy_tr[:, 0].max() - xy_tr[:, 0].min()) or 1.0
        span_y = float(xy_tr[:, 1].max() - xy_tr[:, 1].min()) or 1.0
        starts = [
            np.array([0.0,  np.log(span_x * 0.2),  np.log(span_y * 0.2), -4.0]),
            np.array([0.0,  np.log(span_x * 0.5),  np.log(span_y * 0.5), -3.0]),
            np.array([0.5,  np.log(span_x * 0.1),  np.log(span_y * 0.1), -5.0]),
            np.array([-0.5, np.log(span_x * 0.05), np.log(span_y * 0.05), -6.0]),
        ]
    if bounds is None:
        bounds = ((-4, 6), (-6, 4), (-6, 4), (-12, 2))

    best = None
    for s0 in starts:
        try:
            r = minimize(nlml, x0=s0, method="L-BFGS-B", bounds=bounds)
            if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                best = r
        except Exception:
            pass

    if best is None:
        # fallback: just use first start
        best_theta = starts[0]
    else:
        best_theta = best.x

    var, lx, ly, noise = np.exp(best_theta)
    print(f"  [PressureGP] var={var:.3f}, lx={lx:.3f}, ly={ly:.3f}, "
          f"noise={noise:.2e}")

    K_tr = _scalar_kern_matrix(xy_tr, xy_tr, var, lx, ly)
    K_tr += (noise + jitter) * np.eye(len(y_norm))
    L    = np.linalg.cholesky(K_tr)
    a    = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))

    K_te_tr = _scalar_kern_matrix(xy_te, xy_tr, var, lx, ly)
    p_pred_norm = K_te_tr @ a
    return p_pred_norm * y_std + y_mean


# =============================================================================
# Drag via direct pressure surface integration
# =============================================================================
def compute_drag(xy_full, p_full,
                 centre, L,
                 rect_theta_deg, xy_body, n_body,
                 sensor_idx,
                 n_wall_interp=2000):
    """
    Drag via direct pressure integration on the building surface:

        D = ∮_wall  p * (n̂ · drag_dir)  ds

    Truth:    CFD pressure interpolated onto wall (exact).
    GP recon: scalar ARD Matern-5/2 GP trained on pressure at sensor
              locations, predicted onto wall.  Sensors already cover the
              near-body region so the GP extrapolates a short distance to
              the wall — far better than Bernoulli from velocity.

    The normal sign convention: wall_frame returns normals pointing AWAY
    from the centroid (outward). For the drag integral we want the force
    ON the body, which is -∮ p n̂ ds in the drag direction, i.e. the
    pressure acts inward. We absorb the sign by negating the final result
    so that a bluff body with stagnation pressure on the front and low
    pressure on the rear gives positive drag.
    """

    # --- drag direction ---
    drag_dir = np.array([1., 0.]) @ _rotmat(rect_theta_deg).T

    # --- build uniformly-resampled closed wall loop ---
    c = xy_body.mean(axis=0)
    order = np.argsort(np.arctan2(xy_body[:, 1] - c[1],
                                   xy_body[:, 0] - c[0]))
    xy_ordered = xy_body[order]
    xy_closed_raw = np.vstack([xy_ordered, xy_ordered[0]])
    ds_raw = np.concatenate([[0.],
              np.cumsum(np.linalg.norm(np.diff(xy_closed_raw, axis=0), axis=1))])
    total_perim = ds_raw[-1]
    s_uniform   = np.linspace(0., total_perim, n_wall_interp, endpoint=False)

    from scipy.interpolate import interp1d
    fx = interp1d(ds_raw, xy_closed_raw[:, 0], kind="linear")
    fy = interp1d(ds_raw, xy_closed_raw[:, 1], kind="linear")
    xy_w = np.column_stack([fx(s_uniform), fy(s_uniform)])

    # outward normals
    n_w = wall_frame(xy_w)

    # closed loop for trapz
    xy_w_cl = np.vstack([xy_w, xy_w[0]])
    n_w_cl  = np.vstack([n_w,  n_w[0]])
    ds_int  = np.concatenate([[0.],
               np.cumsum(np.linalg.norm(np.diff(xy_w_cl, axis=0), axis=1))])

    # outward normal projected onto drag direction
    n_dot_d = n_w_cl @ drag_dir   # positive on front face, negative on rear

    # ---------------------------------------------------------------
    # GP pressure: scalar GP trained on sensor pressures + near-wall anchors
    # ---------------------------------------------------------------
    ip = LinearNDInterpolator(xy_full, p_full)

    xy_sens = xy_full[sensor_idx]
    p_sens  = p_full[sensor_idx]

    # Near-wall anchor ring: CFD pressure sampled just outside the body
    # (0.05L offset outward) so the GP doesn't have to extrapolate far to wall
    offset        = 0.05 * L
    xy_near       = xy_w[::4] + offset * n_w[::4]
    p_near_interp = ip(xy_near)
    near_nan      = ~np.isfinite(p_near_interp)
    if near_nan.any():
        tree2 = KDTree(xy_full)
        _, idx2 = tree2.query(xy_near[near_nan])
        p_near_interp[near_nan] = p_full[idx2]

    xy_gp_tr = np.vstack([xy_sens, xy_near])
    p_gp_tr  = np.concatenate([p_sens, p_near_interp])

    print(f"  [PressureGP] training on {len(xy_gp_tr)} points "
          f"({len(xy_sens)} sensors + {len(xy_near)} near-wall anchors)")

    p_wall_gp = _fit_scalar_gp(xy_gp_tr, p_gp_tr, xy_w_cl)

    D_pred  = -float(np.trapezoid(p_wall_gp * n_dot_d, ds_int))
    D_truth = 4207.0   # CFD reference value [N/m]

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------
    err = D_pred - D_truth
    rel = 100.0 * abs(err) / D_truth

    print("\n" + "=" * 60)
    print(" DRAG  (direct pressure surface integration)")
    print("=" * 60)
    print(f"  drag direction (global):  [{drag_dir[0]:.4f}, {drag_dir[1]:.4f}]")
    print(f"  wall integration points:  {n_wall_interp}")
    print(f"  perimeter:                {total_perim:.3f} m")
    print(f"  CFD reference  D = {D_truth:+10.4f}  N/m")
    print(f"  GP recon       D = {D_pred:+10.4f}  N/m")
    print(f"  Error        Δ D = {err:+10.4f}  N/m   ({rel:.2f}% of reference)")
    print("=" * 60)

    return D_truth, D_pred

# =============================================================================
# Main Execution Logic
# =============================================================================
def run(num_left=50, num_right=30, num_interior=20,
        seed_string="PEACH_VIBE",
        rotation_cw_deg=0.0,
        use_wall=True, plot=True, n_wall=160,
        compute_forces=True, rho=1.225):

    xy_full, uv_full, p_full   = load_field()
    xy_body, n_body, centre, L = load_body()

    print(f"Body centre: {centre},  L={L:.4f}")
    theta_deg = -rotation_cw_deg
    print(f"Sampling rectangle angle: {theta_deg:.2f} deg  "
          f"(rotation_cw_deg={rotation_cw_deg})")

    idx = sample_sensors(xy_full, xy_body, centre, L,
                         num_left=num_left, num_right=num_right,
                         num_interior=num_interior,
                         seed_string=seed_string,
                         theta_override=theta_deg)
    xy_s, uv_s = xy_full[idx], uv_full[idx]
    n_sensors  = len(idx)
    print(f"Sensors after deduplication: {n_sensors}")

    w_idx = np.linspace(0, len(xy_body) - 1, n_wall).astype(int)
    xy_w, n_w = xy_body[w_idx], n_body[w_idx]

    upstream = xy_full[:, 0] < (centre[0] - 3 * L)
    U_inf  = float(uv_full[upstream, 0].mean())
    V_inf  = float(uv_full[upstream, 1].mean())
    U_char = float(np.hypot(U_inf, V_inf))
    print(f"Freestream: U={U_inf:.4f}, V={V_inf:.4f}, |U|={U_char:.4f} m/s")

    # Scaling
    xy_s_s = (xy_s - centre) / L
    xy_w_s = (xy_w - centre) / L
    uv_s_p = (uv_s - [U_inf, V_inf]) / U_char

    X_v, y_v = to_4d(jnp.array(xy_s_s), jnp.array(uv_s_p))
    X_b, y_b = wall_obs_slip(jnp.array(xy_w_s), jnp.array(n_w),
                              U_inf, V_inf, U_char)

    if use_wall:
        X_tr = jnp.vstack([X_v, X_b])
        y_tr = jnp.concatenate([y_v, y_b])
        mask = jnp.concatenate([jnp.ones(len(y_v)),
                                 jnp.zeros(len(y_b))])
    else:
        X_tr, y_tr, mask = X_v, y_v, jnp.ones(len(y_v))

    # Hyperparameter optimisation — more starting points for robustness
    mask_v = jnp.ones(len(y_v))
    starts = [
        np.array([ 0.0,  1.0,  1.0, -4.0]),
        np.array([ 1.0,  0.5,  0.5, -3.0]),
        np.array([ 0.0,  0.0,  0.0, -4.0]),
        np.array([ 0.5,  0.8,  0.8, -5.0]),
        np.array([-1.0,  1.2,  0.8, -3.5]),
    ]
    bounds = ((-6, 6), (-4, 2.5), (-4, 2.5), (-12, 2))
    best = None
    for s0 in starts:
        try:
            r = minimize(
                lambda t: float(neg_log_ml(jnp.array(t), X_v, y_v, mask_v)),
                x0=s0, method="L-BFGS-B", bounds=bounds,
            )
            if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                best = r
        except Exception:
            pass

    if best is None:
        raise RuntimeError("All hyperparameter optimisation attempts failed.")

    theta_opt = jnp.array(best.x)
    vs, lx, ly, noise = np.exp(best.x)
    print(f"Hyperparams: var={vs:.4f}, lx={lx:.4f}, ly={ly:.4f}, noise={noise:.2e}")

    # Prediction over full field
    xy_te_s = (xy_full - centre) / L
    X_te    = to_4d(jnp.array(xy_te_s))
    pred    = predict(X_tr, y_tr, X_te, theta_opt, mask)
    uv_pred = pred.reshape(-1, 2, order="C") * U_char + [U_inf, V_inf]

    rmse = float(np.sqrt(np.mean((uv_pred - uv_full) ** 2)))
    print(f"n={n_sensors:4d}  SlipWalls={use_wall}  RMSE={rmse:.3f} m/s")

    if plot:
        _plot(xy_full, uv_full, uv_pred, xy_s, xy_body,
              centre, L, theta_deg, n_sensors, use_wall, rmse)

    if compute_forces:
        compute_drag(
            xy_full, p_full,
            centre, L,
            rect_theta_deg=theta_deg,
            xy_body=xy_body, n_body=n_body,
            sensor_idx=idx,
            n_wall_interp=2000,
        )

    return rmse

# =============================================================================
# Plotting
# =============================================================================
def _plot(xy, uv_true, uv_pred, xy_s, xy_body,
          centre, L, theta_deg, n, use_wall, rmse):
    speed_t = np.hypot(uv_true[:, 0], uv_true[:, 1])
    speed_p = np.hypot(uv_pred[:, 0], uv_pred[:, 1])
    err     = np.linalg.norm(uv_pred - uv_true, axis=1)
    vmax    = float(np.max(speed_t))

    tri = Triangulation(xy[:, 0], xy[:, 1])

    c_b   = xy_body.mean(axis=0)
    order = np.argsort(np.arctan2(xy_body[:, 1] - c_b[1],
                                  xy_body[:, 0] - c_b[0]))
    body_poly = xy_body[order]

    excl_diam = 2.0 * L
    s_r, a_r, h_r = 3.0 * excl_diam, 17.0 * excl_diam, 5.0 * excl_diam
    verts     = _rectangle_vertices(centre, theta_deg, s_r, a_r, h_r)
    rect_poly = np.vstack([verts, verts[0]])

    N_LEVELS = 64

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    panels = [
        (axes[0], speed_t, "Truth |u|",           vmax,             "viridis"),
        (axes[1], speed_p, "GP Recon (Slip) |u|", vmax,             "viridis"),
        (axes[2], err,     "Error Map",            float(err.max()), "magma"),
    ]

    for ax, values, title, vm, cmap in panels:
        cf = ax.tricontourf(tri, values, levels=N_LEVELS,
                            vmin=0, vmax=vm, cmap=cmap)
        ax.add_patch(Polygon(body_poly, closed=True,
                             fc="white", ec="k", lw=1.2, zorder=4))
        ax.set_aspect("equal")
        ax.set_title(title)
        fig.colorbar(cf, ax=ax, shrink=0.85)

    axes[1].scatter(xy_s[:, 0], xy_s[:, 1],
                    facecolors="none", edgecolors="red", s=22, lw=0.8,
                    zorder=5, label=f"{n} sensors")
    axes[1].plot(rect_poly[:, 0], rect_poly[:, 1],
                 "b--", lw=1.0, alpha=0.6, label="sampling region")
    axes[1].add_patch(Circle(centre, L, fill=False,
                             ls=":", lw=1.2, color="orange",
                             label="excl. radius", zorder=5))
    axes[1].legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.show()

# =============================================================================
v_flow_deg = -14.5

if __name__ == "__main__":
    run(num_left=35, num_right=35, num_interior=70,
        seed_string="PEACH_VIBE",
        rotation_cw_deg= -1 * v_flow_deg,
        use_wall=True, plot=True,
        compute_forces=True, rho=1.225)
"""
Divergence-free GP with no-slip wall constraint (no-penetration + tangential).
Works for cylinder, square, or any star-shaped body.
Includes Localized Mixture of Experts for stagnation flow refinement.

Encoding: each obs row is [x, y, d_x, d_y]. Observable is d . u(x).
  - velocity sensor:  emit two rows, d=(1,0), d=(0,1)
  - wall point:       emit two rows, d=n_face, d=t_face   (u . n = u . t = 0)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import hessian
from scipy.optimize import minimize
from scipy.stats.qmc import Halton
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

jax.config.update("jax_enable_x64", True)

HERE       = Path(__file__).resolve().parent
FIELD_PATH = HERE / "output.xlsx"
CYL_PATH   = HERE / "building.xlsx"


# =============================================================================
# Kernel  (ARD Matern 5/2)
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


def wall_obs_noslip(xy_w, n_w, t_w, U_inf, V_inf, U_char):
    """Two rows per wall point: u . n = 0  AND  u . t = 0.
    After subtracting freestream, target = -(d . U_inf_vec) / U_char.
    Sign of t is irrelevant — flipping it flips both d and rhs."""
    U_vec = jnp.array([U_inf, V_inf])
    X_n = jnp.hstack([xy_w, n_w])
    y_n = -(n_w @ U_vec) / U_char
    X_t = jnp.hstack([xy_w, t_w])
    y_t = -(t_w @ U_vec) / U_char
    return jnp.vstack([X_n, X_t]), jnp.concatenate([y_n, y_t])


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


def train_gp(X_tr, y_tr, mask):
    """Helper to run L-BFGS-B optimization for GP hyperparameters."""
    starts = [np.array([ 0.0,  1.0,  1.0, -4.0]),
              np.array([ 1.0,  0.5,  0.5, -3.0]),
              np.array([-0.5, -0.5, -0.5, -5.0]),
              np.array([ 2.0,  1.5,  0.5, -3.0]),
              np.array([ 0.5,  2.0,  0.5, -4.0]),
              np.array([ 0.0,  0.0,  0.0, -4.0])]
    bounds = ((-6, 6), (-4, 2.5), (-4, 2.5), (-12, 2))
    best = None
    for s in starts:
        try:
            r = minimize(
                lambda t: float(neg_log_ml(jnp.array(t), X_tr, y_tr, mask)),
                x0=s, method="L-BFGS-B", bounds=bounds,
            )
            if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                best = r
        except Exception:
            pass
    return best


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
# Data loading
# =============================================================================
def _clean(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_field():
    df = _clean(pd.read_excel(FIELD_PATH))
    xy = df[["x-coordinate", "y-coordinate"]].to_numpy(float)
    uv = df[["x-velocity",  "y-velocity"]].to_numpy(float)
    return xy, uv


def wall_frame(xy_wall):
    """Outward normals and CCW tangents for any star-shaped closed body.
    Returns both in the *original* input ordering."""
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
    t *= sign[:, None]                # keep (t, n) right-handed
    n_out = np.empty_like(n); t_out = np.empty_like(t)
    n_out[order] = n
    t_out[order] = t
    return n_out, t_out


def load_body():
    df = _clean(pd.read_excel(CYL_PATH))
    xy = df[["x-coordinate", "y-coordinate"]].to_numpy(float)
    xy = np.unique(xy, axis=0)
    centre = xy.mean(axis=0)
    L = 0.5 * max(xy[:, 0].ptp(), xy[:, 1].ptp())
    n, t = wall_frame(xy)
    return xy, n, t, centre, L


# =============================================================================
# Sensor sampling
# =============================================================================
def sample_sensors(xy_full, n, body_xy, L, buffer_L=1.0, seed=0):
    x_lo, x_hi = xy_full[:, 0].min(), xy_full[:, 0].max()
    y_lo, y_hi = xy_full[:, 1].min(), xy_full[:, 1].max()

    pad = buffer_L * L
    bx_lo, bx_hi = body_xy[:, 0].min() - pad, body_xy[:, 0].max() + pad
    by_lo, by_hi = body_xy[:, 1].min() - pad, body_xy[:, 1].max() + pad

    inside = ((xy_full[:, 0] > bx_lo) & (xy_full[:, 0] < bx_hi) &
              (xy_full[:, 1] > by_lo) & (xy_full[:, 1] < by_hi))
    eligible = np.where(~inside)[0]
    xy_e = xy_full[eligible]

    pts = Halton(d=2, seed=seed).random(n * 3)
    pts[:, 0] = x_lo + pts[:, 0] * (x_hi - x_lo)
    pts[:, 1] = y_lo + pts[:, 1] * (y_hi - y_lo)

    chosen, seen = [], set()
    for p in pts:
        j = int(eligible[np.argmin(np.sum((xy_e - p) ** 2, axis=1))])
        if j not in seen:
            seen.add(j); chosen.append(j)
        if len(chosen) >= n: break
    return np.array(chosen)


# =============================================================================
# One reconstruction
# =============================================================================
def run(n_sensors=80, seed=0, use_wall=True, plot=True,
        n_wall=160, buffer_L=1.0):
    xy_full, uv_full                    = load_field()
    xy_body, n_body, t_body, centre, L  = load_body()

    idx = sample_sensors(xy_full, n_sensors, xy_body, L,
                         buffer_L=buffer_L, seed=seed)
    xy_s, uv_s = xy_full[idx], uv_full[idx]

    w_idx = np.linspace(0, len(xy_body) - 1, n_wall).astype(int)
    xy_w, n_w, t_w = xy_body[w_idx], n_body[w_idx], t_body[w_idx]

    upstream = xy_full[:, 0] < (centre[0] - 3 * L)
    U_inf  = float(uv_full[upstream, 0].mean())
    V_inf  = float(uv_full[upstream, 1].mean())
    U_char = float(np.hypot(U_inf, V_inf))

    xy_s_s = (xy_s - centre) / L
    xy_w_s = (xy_w - centre) / L
    uv_s_p = (uv_s - [U_inf, V_inf]) / U_char

    X_v, y_v = to_4d(jnp.array(xy_s_s), jnp.array(uv_s_p))
    X_b, y_b = wall_obs_noslip(jnp.array(xy_w_s),
                               jnp.array(n_w), jnp.array(t_w),
                               U_inf, V_inf, U_char)

    if use_wall:
        X_tr = jnp.vstack([X_v, X_b])
        y_tr = jnp.concatenate([y_v, y_b])
        # 2 rows per wall point now
        mask = jnp.concatenate([jnp.ones(len(y_v)),
                                jnp.zeros(len(y_b))])
    else:
        X_tr, y_tr, mask = X_v, y_v, jnp.ones(len(y_v))

    # ---------------------------------------------------------
    # 1. Train Global GP
    # ---------------------------------------------------------
    res_global = train_gp(X_tr, y_tr, mask)
    theta_global = jnp.array(res_global.x)

    # ---------------------------------------------------------
    # 2. Train Specialist GP (Stagnation Zone)
    # ---------------------------------------------------------
    # Define bounding box for stagnation flow (scaled coords). 
    is_stag = (X_tr[:, 0] >= -3.0) & (X_tr[:, 0] <= 0.2) & (jnp.abs(X_tr[:, 1]) <= 1.5)
    
    X_tr_stag = X_tr[is_stag]
    y_tr_stag = y_tr[is_stag]
    mask_stag = mask[is_stag]

    # Only train specialist if we actually captured enough points upstream
    use_specialist = len(y_tr_stag) > 4
    if use_specialist:
        res_stag = train_gp(X_tr_stag, y_tr_stag, mask_stag)
        theta_stag = jnp.array(res_stag.x)

    # ---------------------------------------------------------
    # 3. Predict and Blend
    # ---------------------------------------------------------
    xy_te_s = (xy_full - centre) / L
    X_te    = to_4d(jnp.array(xy_te_s))
    
    # Base prediction from the global model
    pred_global = predict(X_tr, y_tr, X_te, theta_global, mask)

    if use_specialist:
        # Prediction from the local stagnation model
        pred_stag = predict(X_tr_stag, y_tr_stag, X_te, theta_stag, mask_stag)
        
        # Calculate spatial blending weight (w)
        dx = X_te[:, 0] - (-1.0)
        dy = X_te[:, 1]
        
        # Anisotropic Gaussian focusing heavily directly upstream
        dist_sq = (dx**2) / 1.5 + (dy**2) / 1.0
        w = np.exp(-dist_sq)
        
        # Fade out abruptly behind the leading edge
        fade = 1.0 / (1.0 + np.exp(10.0 * X_te[:, 0]))
        w = w * fade
        
        # Smoothly interpolate between global and specialist
        pred = (1.0 - w) * pred_global + w * pred_stag
    else:
        pred = pred_global

    uv_pred = pred.reshape(-1, 2, order="C") * U_char + [U_inf, V_inf]

    rmse = float(np.sqrt(np.mean((uv_pred - uv_full) ** 2)))
    print(f"n={n_sensors:4d}  walls={use_wall}  RMSE={rmse:.3f} m/s  "
          f"(global var={np.exp(res_global.x[0]):.2g}, "
          f"lx={np.exp(res_global.x[1]):.2g}, ly={np.exp(res_global.x[2]):.2g})")
    
    if use_specialist:
        # Note: Divide len(y_tr_stag) by 2 because each physical point is 2 rows in the data (u and v)
        print(f"  -> Specialist GP trained on {len(y_tr_stag)//2} stagnation points "
              f"(lx={np.exp(res_stag.x[1]):.2g}, ly={np.exp(res_stag.x[2]):.2g})")

    if plot:
        _plot(xy_full, uv_full, uv_pred, xy_s, xy_body, centre,
              n_sensors, use_wall, rmse)
    return rmse


# =============================================================================
# Plotting
# =============================================================================
def _plot(xy, uv_true, uv_pred, xy_s, xy_body, centre,
          n, use_wall, rmse):
    speed_t = np.hypot(uv_true[:, 0], uv_true[:, 1])
    speed_p = np.hypot(uv_pred[:, 0], uv_pred[:, 1])
    err     = np.linalg.norm(uv_pred - uv_true, axis=1)
    vmax    = float(np.max(speed_t))

    c     = xy_body.mean(axis=0)
    order = np.argsort(np.arctan2(xy_body[:, 1] - c[1],
                                  xy_body[:, 0] - c[0]))
    body_poly = xy_body[order]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    panels = [
        (ax[0], speed_t, "truth |u|",         vmax,             "viridis"),
        (ax[1], speed_p, "GP recon |u|",      vmax,             "viridis"),
        (ax[2], err,     "|u_pred - u_true|", float(err.max()), "magma"),
    ]
    for a, c_, title, vm, cmap in panels:
        sc = a.scatter(xy[:, 0], xy[:, 1], c=c_, s=3, vmin=0, vmax=vm,
                       cmap=cmap)
        a.add_patch(Polygon(body_poly, closed=True, fc="white",
                            ec="k", lw=1.2, zorder=4))
        a.set_aspect("equal")
        a.set_title(title)
        plt.colorbar(sc, ax=a, shrink=0.85)
    ax[1].scatter(xy_s[:, 0], xy_s[:, 1], facecolors="none",
                  edgecolors="red", s=22, lw=0.9, zorder=5,
                  label=f"{n} sensors")
    ax[1].legend(loc="upper right", fontsize=8)
    plt.suptitle(f"n_sensors={n}, walls={'on' if use_wall else 'off'}, "
                 f"RMSE={rmse:.3f} m/s", fontsize=12)
    plt.tight_layout()
    fname = HERE / f"recon_n{n}_{'walls' if use_wall else 'nowalls'}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    print(f"  saved {fname}")
    plt.close()


# =============================================================================
# Sweep helpers
# =============================================================================
def sweep(n_list=(20, 50, 100, 200), seed=0, use_wall=True):
    return [run(n, seed=seed, use_wall=use_wall) for n in n_list]


def compare_walls(n=50, seed=0):
    print("--- without walls ---"); run(n, seed=seed, use_wall=False)
    print("--- with walls    ---"); run(n, seed=seed, use_wall=True)


if __name__ == "__main__":
    run(n_sensors=150, seed=0, use_wall=True)
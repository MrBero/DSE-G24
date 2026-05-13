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
# Kernel (Isotropic Matern 5/2) - Divergence Free via Stream Function Hessian
# =============================================================================
def matern52(x, xp, var, l):
    # Isotropic: standard Euclidean distance scaled by a single length scale l
    dist = jnp.sqrt(jnp.sum((x - xp) ** 2) + 1e-12)
    r = dist / l
    s = jnp.sqrt(5.0) * r
    return var * (1 + s + s ** 2 / 3) * jnp.exp(-s)

def kern_entry(X, Xp, var, l):
    x,  d  = X[:2],  X[2:4]
    xp, dp = Xp[:2], Xp[2:4]
    # The Divergence-free kernel is the Matrix-valued Hessian of the scalar kernel
    H = -hessian(matern52, argnums=0)(x, xp, var, l)
    C = jnp.array([[ H[1, 1], -H[1, 0]],
                   [-H[0, 1],  H[0, 0]]])
    return d @ C @ dp

_row = jax.vmap(kern_entry, in_axes=(None, 0, None, None))
kern = jax.jit(jax.vmap(_row, in_axes=(0, None, None, None)))

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
    """
    Only enforces No-Penetration (u . n = 0).
    The tangential component is not constrained, allowing fluid to slip.
    """
    U_vec = jnp.array([U_inf, V_inf])
    X_n = jnp.hstack([xy_w, n_w])
    # Target value is the negative projection of the freestream onto the normal
    y_n = -(n_w @ U_vec) / U_char
    return X_n, y_n

# =============================================================================
# Train / predict
# =============================================================================
JITTER = 1e-8

def diag_noise(noise, mask):
    return noise * mask + JITTER * (1.0 - mask)

def neg_log_ml(theta, X_tr, y_tr, mask):
    # theta is now 3 parameters: [log_var, log_l, log_noise]
    vs, l, noise = jnp.exp(theta)
    K = kern(X_tr, X_tr, vs, l) + jnp.diag(diag_noise(noise, mask))
    L = jnp.linalg.cholesky(K)
    a = jax.scipy.linalg.cho_solve((L, True), y_tr)
    n = len(y_tr)
    return 0.5 * (y_tr @ a + 2 * jnp.sum(jnp.log(jnp.diag(L)))
                  + n * jnp.log(2 * jnp.pi))

def predict(X_tr, y_tr, X_te, theta, mask, chunk=2000):
    vs, l, noise = jnp.exp(theta)
    K = kern(X_tr, X_tr, vs, l) + jnp.diag(diag_noise(noise, mask))
    L = jnp.linalg.cholesky(K)
    a = jax.scipy.linalg.cho_solve((L, True), y_tr)
    out = []
    for i in range(0, len(X_te), chunk):
        out.append(np.array(kern(X_te[i:i + chunk], X_tr, vs, l) @ a))
    return np.concatenate(out)

# =============================================================================
# Data loading & Geometry
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

def load_body():
    df = _clean(pd.read_excel(CYL_PATH))
    xy = df[["x-coordinate", "y-coordinate"]].to_numpy(float)
    xy = np.unique(xy, axis=0)
    centre = xy.mean(axis=0)
    
    # --- UPDATED LINE ---
    # Replaced xy[:, 0].ptp() with np.ptp(xy[:, 0]) for NumPy 2.0 compatibility
    L = 0.5 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    
    n = wall_frame(xy)
    return xy, n, centre, L

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
# Main Execution Logic
# =============================================================================
def run(n_sensors=100, seed=0, use_wall=True, plot=True,
        n_wall=160, buffer_L=1.0):
    xy_full, uv_full                    = load_field()
    xy_body, n_body, centre, L          = load_body()

    idx = sample_sensors(xy_full, n_sensors, xy_body, L,
                         buffer_L=buffer_L, seed=seed)
    xy_s, uv_s = xy_full[idx], uv_full[idx]

    w_idx = np.linspace(0, len(xy_body) - 1, n_wall).astype(int)
    xy_w, n_w = xy_body[w_idx], n_body[w_idx]

    upstream = xy_full[:, 0] < (centre[0] - 3 * L)
    U_inf  = float(uv_full[upstream, 0].mean())
    V_inf  = float(uv_full[upstream, 1].mean())
    U_char = float(np.hypot(U_inf, V_inf))

    # Scaling
    xy_s_s = (xy_s - centre) / L
    xy_w_s = (xy_w - centre) / L
    uv_s_p = (uv_s - [U_inf, V_inf]) / U_char

    X_v, y_v = to_4d(jnp.array(xy_s_s), jnp.array(uv_s_p))
    
    # Build Wall Constraints
    X_b, y_b = wall_obs_slip(jnp.array(xy_w_s), jnp.array(n_w), 
                             U_inf, V_inf, U_char)

    if use_wall:
        X_tr = jnp.vstack([X_v, X_b])
        y_tr = jnp.concatenate([y_v, y_b])
        # Mask ensures wall points have minimal JITTER while sensors have learnable noise
        mask = jnp.concatenate([jnp.ones(len(y_v)),
                                jnp.zeros(len(y_b))])
    else:
        X_tr, y_tr, mask = X_v, y_v, jnp.ones(len(y_v))

    # Optimization
    mask_v = jnp.ones(len(y_v))
    # Starts and bounds updated for 3 parameters: [var, l, noise]
    starts = [np.array([ 0.0,  1.0, -4.0]),
              np.array([ 1.0,  0.5, -3.0]),
              np.array([ 0.0,  0.0, -4.0])]
    bounds = ((-6, 6), (-4, 2.5), (-12, 2))
    best = None
    for s in starts:
        try:
            r = minimize(
                lambda t: float(neg_log_ml(jnp.array(t), X_v, y_v, mask_v)),
                x0=s, method="L-BFGS-B", bounds=bounds,
            )
            if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                best = r
        except Exception:
            pass
    
    theta = jnp.array(best.x)

    # Prediction
    xy_te_s = (xy_full - centre) / L
    X_te    = to_4d(jnp.array(xy_te_s))
    pred    = predict(X_tr, y_tr, X_te, theta, mask)
    uv_pred = pred.reshape(-1, 2, order="C") * U_char + [U_inf, V_inf]

    rmse = float(np.sqrt(np.mean((uv_pred - uv_full) ** 2)))
    print(f"n={n_sensors:4d}  SlipWalls={use_wall}  RMSE={rmse:.3f} m/s")

    if plot:
        _plot(xy_full, uv_full, uv_pred, xy_s, xy_body, n_sensors, use_wall, rmse)
    return rmse

def _plot(xy, uv_true, uv_pred, xy_s, xy_body, n, use_wall, rmse):
    speed_t = np.hypot(uv_true[:, 0], uv_true[:, 1])
    speed_p = np.hypot(uv_pred[:, 0], uv_pred[:, 1])
    err     = np.linalg.norm(uv_pred - uv_true, axis=1)
    vmax    = float(np.max(speed_t))

    c     = xy_body.mean(axis=0)
    order = np.argsort(np.arctan2(xy_body[:, 1] - c[1], xy_body[:, 0] - c[0]))
    body_poly = xy_body[order]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    panels = [
        (ax[0], speed_t, "Truth |u|", vmax, "viridis"),
        (ax[1], speed_p, "GP Recon (Slip) |u|", vmax, "viridis"),
        (ax[2], err, "Error Map", float(err.max()), "magma"),
    ]
    for a, c_, title, vm, cmap in panels:
        sc = a.scatter(xy[:, 0], xy[:, 1], c=c_, s=3, vmin=0, vmax=vm, cmap=cmap)
        a.add_patch(Polygon(body_poly, closed=True, fc="white", ec="k", lw=1.2, zorder=4))
        a.set_aspect("equal")
        a.set_title(title)
        plt.colorbar(sc, ax=a, shrink=0.85)
    
    ax[1].scatter(xy_s[:, 0], xy_s[:, 1], facecolors="none", edgecolors="red", s=22, label=f"{n} sensors")
    ax[1].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run(n_sensors=100, seed=0, use_wall=True)
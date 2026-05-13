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

import gpjax as gpx
from gpjax.kernels import AbstractKernel

jax.config.update("jax_enable_x64", True)

HERE       = Path(__file__).resolve().parent
FIELD_PATH = HERE / "output.xlsx"
CYL_PATH   = HERE / "building.xlsx"

# =============================================================================
# Kernel (ARD Matern 5/2) - Divergence Free using GPJax AbstractKernel
# =============================================================================
class DivFreeMatern52(AbstractKernel):
    def __init__(self, variance: float, lengthscale_x: float, lengthscale_y: float):
        super().__init__()
        self.variance = variance
        self.lengthscale_x = lengthscale_x
        self.lengthscale_y = lengthscale_y

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        var = jnp.exp(self.variance)
        lx  = jnp.exp(self.lengthscale_x)
        ly  = jnp.exp(self.lengthscale_y)

        def _matern52(x_coords, y_coords):
            dx = (x_coords[0] - y_coords[0]) / lx
            dy = (x_coords[1] - y_coords[1]) / ly
            r  = jnp.sqrt(dx * dx + dy * dy + 1e-12)
            s  = jnp.sqrt(5.0) * r
            return var * (1 + s + s ** 2 / 3) * jnp.exp(-s)

        pos_x, dir_x = x[:2], x[2:4]
        pos_y, dir_y = y[:2], y[2:4]

        H = -hessian(_matern52, argnums=0)(pos_x, pos_y)
        C = jnp.array([[ H[1, 1], -H[1, 0]],
                       [-H[0, 1],  H[0, 0]]])
        return jnp.dot(dir_x, jnp.dot(C, dir_y))

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
    U_vec = jnp.array([U_inf, V_inf])
    X_n = jnp.hstack([xy_w, n_w])
    y_n = -(n_w @ U_vec) / U_char
    return X_n, y_n

# =============================================================================
# Train / predict
# =============================================================================
JITTER = 1e-5 

@jax.jit
def neg_log_ml(theta, X_tr, y_tr, mask):
    kernel = DivFreeMatern52(
        variance=theta[0], 
        lengthscale_x=theta[1], 
        lengthscale_y=theta[2]
    )
    
    K = kernel.gram(X_tr)
    if hasattr(K, "to_dense"):
        K = K.to_dense()

    noise_diag = jnp.exp(theta[3]) * mask + JITTER * (1.0 - mask)
    K = K + jnp.diag(noise_diag)
    
    L = jnp.linalg.cholesky(K)
    a = jax.scipy.linalg.cho_solve((L, True), y_tr)
    n = len(y_tr)
    return 0.5 * (jnp.dot(y_tr, a) + 2 * jnp.sum(jnp.log(jnp.diag(L))) + n * jnp.log(2 * jnp.pi))

def predict(X_tr, y_tr, X_te, theta, mask, chunk=2000):
    kernel = DivFreeMatern52(
        variance=theta[0], 
        lengthscale_x=theta[1], 
        lengthscale_y=theta[2]
    )
    
    K_tr = kernel.gram(X_tr)
    if hasattr(K_tr, "to_dense"):
        K_tr = K_tr.to_dense()

    noise_diag = jnp.exp(theta[3]) * mask + JITTER * (1.0 - mask)
    K_tr = K_tr + jnp.diag(noise_diag)
    
    L = jnp.linalg.cholesky(K_tr)
    a = jax.scipy.linalg.cho_solve((L, True), y_tr)
    
    out = []
    for i in range(0, len(X_te), chunk):
        K_cross = kernel.cross_covariance(X_te[i:i + chunk], X_tr)
        if hasattr(K_cross, "to_dense"):
            K_cross = K_cross.to_dense()
        out.append(np.array(jnp.dot(K_cross, a)))
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
    c = xy_wall.mean(axis=0)
    order = np.argsort(np.arctan2(xy_wall[:, 1] - c[1], xy_wall[:, 0] - c[0]))
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
    L = 0.5 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    n = wall_frame(xy)
    return xy, n, centre, L

# =============================================================================
# Sensor sampling
# =============================================================================
def sample_interior(xy_full, n, body_xy, L, buffer_L=1.5, seed=0):
    """Uniformly samples SPATIAL points, then snaps to nearest mesh node."""
    if n == 0: return np.array([], dtype=int)
    
    x_lo, x_hi = xy_full[:, 0].min(), xy_full[:, 0].max()
    y_lo, y_hi = xy_full[:, 1].min(), xy_full[:, 1].max()

    pad = buffer_L * L
    bx_lo, bx_hi = body_xy[:, 0].min() - pad, body_xy[:, 0].max() + pad
    by_lo, by_hi = body_xy[:, 1].min() - pad, body_xy[:, 1].max() + pad

    # Identify eligible mesh points safely OUTSIDE the exclusion zone
    inside = ((xy_full[:, 0] > bx_lo) & (xy_full[:, 0] < bx_hi) &
              (xy_full[:, 1] > by_lo) & (xy_full[:, 1] < by_hi))
    eligible = np.where(~inside)[0]
    xy_e = xy_full[eligible]

    # Generate a large pool of SPATIALLY uniform points (Halton sequence)
    # We generate extra (n*10) to account for discarded points inside the exclusion zone
    pts = Halton(d=2, seed=seed).random(n * 10)
    pts[:, 0] = x_lo + pts[:, 0] * (x_hi - x_lo)
    pts[:, 1] = y_lo + pts[:, 1] * (y_hi - y_lo)

    # Filter out spatial points that fall inside the building exclusion zone
    spat_inside = ((pts[:, 0] > bx_lo) & (pts[:, 0] < bx_hi) &
                   (pts[:, 1] > by_lo) & (pts[:, 1] < by_hi))
    valid_pts = pts[~spat_inside]

    chosen, seen = [], set()
    for p in valid_pts:
        # Snap the spatially uniform point to the nearest eligible CFD mesh node
        j = int(eligible[np.argmin(np.sum((xy_e - p) ** 2, axis=1))])
        if j not in seen:
            seen.add(j)
            chosen.append(j)
        if len(chosen) >= n: break

    return np.array(chosen)

def sample_lines(xy_full, n_pts_per_line=25):
    """Samples exact lines of sensors spanning the ENTIRE Y-height at X_start and X=250."""
    x_min = xy_full[:, 0].min()
    targets = [x_min, 250.0]
    
    chosen_idx = []
    
    for target in targets:
        closest_x = xy_full[np.argmin(np.abs(xy_full[:, 0] - target)), 0]
        
        tol = 0.1 
        line_idx = np.where(np.abs(xy_full[:, 0] - closest_x) <= tol)[0]
        
        sorted_line_idx = line_idx[np.argsort(xy_full[line_idx, 1])]
        
        if len(sorted_line_idx) > n_pts_per_line:
            even_indices = np.linspace(0, len(sorted_line_idx) - 1, n_pts_per_line).astype(int)
            selected = sorted_line_idx[even_indices]
        else:
            selected = sorted_line_idx
            
        chosen_idx.append(selected)
        
    return np.concatenate(chosen_idx)

# =============================================================================
# Main Execution Logic
# =============================================================================
def run(n_interior=100, n_line_pts=25, seed=0, use_wall=True, plot=True,
        n_wall=160, buffer_L=1.5):
    xy_full, uv_full                    = load_field()
    xy_body, n_body, centre, L          = load_body()

    # Get interior random spatially uniform sensors
    idx_interior = sample_interior(xy_full, n_interior, xy_body, L,
                                   buffer_L=buffer_L, seed=seed)
    
    # Get the targeted line sensors at x_min and x=250 evenly spaced in Y
    idx_lines = sample_lines(xy_full, n_pts_per_line=n_line_pts)
    
    # Combine both sets uniquely
    idx = np.unique(np.concatenate([idx_interior, idx_lines]).astype(int))
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
        mask = jnp.concatenate([jnp.ones(len(y_v)),
                                jnp.zeros(len(y_b))])
    else:
        X_tr, y_tr, mask = X_v, y_v, jnp.ones(len(y_v))

    # Optimization
    mask_v = jnp.ones(len(y_v))
    starts = [np.array([ 0.0,  1.0,  1.0, -4.0]),
              np.array([ 1.0,  0.5,  0.5, -3.0]),
              np.array([ 0.0,  0.0,  0.0, -4.0])]
    bounds = ((-6, 6), (-4, 2.5), (-4, 2.5), (-12, 2))
    best = None
    
    for i, s in enumerate(starts):
        try:
            r = minimize(
                lambda t: float(neg_log_ml(jnp.array(t), X_v, y_v, mask_v)),
                x0=s, method="L-BFGS-B", bounds=bounds,
            )
            if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                best = r
        except Exception as e:
            print(f"Optimization start {i} failed: {e}")
            
    if best is None:
        print("Warning: All optimization restarts failed! Using default starting parameters.")
        theta = jnp.array(starts[0]) 
    else:
        theta = jnp.array(best.x)

    # Prediction
    xy_te_s = (xy_full - centre) / L
    X_te    = to_4d(jnp.array(xy_te_s))
    pred    = predict(X_tr, y_tr, X_te, theta, mask)
    uv_pred = pred.reshape(-1, 2, order="C") * U_char + [U_inf, V_inf]

    rmse = float(np.sqrt(np.mean((uv_pred - uv_full) ** 2)))
    
    if plot:
        _plot(xy_full, uv_full, uv_pred, xy_s, xy_body, len(idx), use_wall, rmse)
        
    return rmse

def _plot(xy, uv_true, uv_pred, xy_s, xy_body, n_sensors, use_wall, rmse):
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
        (ax[1], speed_p, f"GP Recon (RMSE: {rmse:.4f})", vmax, "viridis"),
        # Locked Error Map to scale from 0 to 4.0 m/s
        (ax[2], err, "Error Map", 4.0, "magma"),
    ]
    
    for a, c_, title, vm, cmap in panels:
        sc = a.scatter(xy[:, 0], xy[:, 1], c=c_, s=3, vmin=0, vmax=vm, cmap=cmap)
        a.add_patch(Polygon(body_poly, closed=True, fc="white", ec="k", lw=1.2, zorder=4))
        a.set_aspect("equal")
        a.set_title(title)
        plt.colorbar(sc, ax=a, shrink=0.85)
    
    ax[1].scatter(xy_s[:, 0], xy_s[:, 1], facecolors="none", edgecolors="red", s=22, label=f"{n_sensors} sensors")
    ax[1].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    final_rmse = run(n_interior=100, n_line_pts=15, seed=42, use_wall=True, buffer_L=1)
    
    print("\n" + "="*40)
    print(f"FINAL MODEL RMSE: {final_rmse:.5f} m/s")
    print("="*40 + "\n")
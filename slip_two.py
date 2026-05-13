# slip.py  –  GPR flow reconstruction with divergence-free kernel
#             Sensor placement via sampling_region logic, auto-derived from body geometry.

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import hessian
from scipy.optimize import minimize
from scipy.spatial import KDTree
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
    """
    Read a data file robustly regardless of whether it is:
      - a valid .xlsx  (openpyxl)
      - an old .xls    (xlrd)
      - a CSV saved with a .xlsx extension
    """
    path = Path(path)
    # --- try xlsx first ---
    try:
        return _clean(pd.read_excel(path, engine="openpyxl"))
    except Exception:
        pass
    # --- try old xls format ---
    try:
        return _clean(pd.read_excel(path, engine="xlrd"))
    except Exception:
        pass
    # --- fall back to CSV (handles files mis-saved with .xlsx extension) ---
    for sep in (",", "\t", ";", " "):
        try:
            df = _clean(pd.read_csv(path, sep=sep))
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    raise ValueError(
        f"Cannot read '{path}'. It is not a valid xlsx, xls, or CSV file. "
        "Re-export it from your CFD solver as a proper .xlsx or .csv."
    )

def load_field():
    df = _read_table(FIELD_PATH)
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

def _geometric_diameter(xy):
    """
    Returns the largest straight-line distance between any two boundary points
    — the true geometric diameter of the point cloud.
    Uses rotating calipers on the convex hull: O(N log N).
    For a rectangle this equals the diagonal (hypotenuse of width and height).
    """
    from scipy.spatial import ConvexHull
    if len(xy) < 2:
        return 0.0
    try:
        hull_pts = xy[ConvexHull(xy).vertices]
    except Exception:
        hull_pts = xy          # degenerate: collinear points
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
    # L = half the geometric diameter: the longest line between any two
    # boundary points. For a rectangle this is half the diagonal, i.e.
    # sqrt(width^2 + height^2) / 2.
    L = _geometric_diameter(xy) / 2.0
    n = wall_frame(xy)
    return xy, n, centre, L

# =============================================================================
# Sampling region geometry  (auto-derived from body)
# =============================================================================

def _rotmat(theta_deg):
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def _rectangle_vertices(center, theta_deg, s, a, h):
    """Four corners of the oriented sampling rectangle."""
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
    """
    Derive sampling-region rectangle parameters from the body geometry.
    The rectangle long-axis defaults to horizontal (0 deg, aligned with +x /
    flow direction). Use rotation_cw_deg in run() to tilt it for angled inflow.

      excl_radius  == L  (half max-extent of body)
      excl_diam    == 2 * L
      s            == 3 * excl_diam  (upstream extent)
      a            == 9 * excl_diam  (downstream extent)
      h            == 2 * excl_diam  (half-height)
    """
    excl_diam = 2.0 * L
    s         = 3.0 * excl_diam
    a         = 9.0 * excl_diam
    h         = 2.0 * excl_diam
    return L, s, a, h     # excl_radius == L; angle handled externally


def _seed_from_string(seed_string):
    digest = hashlib.sha256(seed_string.encode("utf-8")).hexdigest()
    return int(digest, 16) % (2 ** 32)


# =============================================================================
# Sensor sampling  (replaces old sample_sensors)
# =============================================================================

def sample_sensors(xy_full, xy_body, centre, L,
                   num_left=50, num_right=30, num_interior=20,
                   seed_string="PEACH_VIBE", theta_override=None):
    """
    Generate sensor positions using the sampling_region strategy, then snap
    each point to the nearest CFD grid node.  Duplicates are removed.

    Parameters
    ----------
    xy_full        : (N,2) all CFD grid points
    xy_body        : (M,2) body surface points
    centre         : (2,)  body centroid
    L              : float half max-extent (used as exclusion radius)
    num_left       : int   points along left (upstream) boundary of rectangle
    num_right      : int   points along right (downstream) boundary
    num_interior   : int   randomly placed interior points
    seed_string    : str   string whose SHA-256 hash seeds the RNG
    theta_override : float if provided, overrides the PCA-derived angle (deg)

    Returns
    -------
    idx : 1-D int array of indices into xy_full
    """
    excl_radius, s, a, h = _derive_sampling_geometry(centre, L)
    if theta_override is not None:
        theta_deg = theta_override
    else:
        theta_deg = 0.0

    seed  = _seed_from_string(seed_string)
    rng   = np.random.default_rng(seed)
    verts = _rectangle_vertices(centre, theta_deg, s, a, h)

    # --- left boundary (upstream edge) ---
    left_pts = np.linspace(verts[0], verts[3], num_left)

    # --- right boundary (downstream edge) ---
    right_pts = np.linspace(verts[1], verts[2], num_right)

    # --- interior points (random, within rectangle, outside exclusion circle) ---
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

    # --- snap each point to nearest CFD node, drop duplicates ---
    tree = KDTree(xy_full)
    _, raw_idx = tree.query(all_pts)
    _, unique_pos = np.unique(raw_idx, return_index=True)
    idx = raw_idx[np.sort(unique_pos)]   # preserve original ordering

    return idx


# =============================================================================
# Main Execution Logic
# =============================================================================
def run(num_left=50, num_right=30, num_interior=20,
        seed_string="PEACH_VIBE",
        rotation_cw_deg=0.0,
        use_wall=True, plot=True, n_wall=160):
    """
    rotation_cw_deg : float
        Additional clockwise rotation (degrees) applied to the sampling
        rectangle on top of the PCA-derived body orientation.
        E.g. set to 14.5 if inflow arrives 14.5 deg from above, so the
        rectangle aligns with the flow direction.
    """
    xy_full, uv_full           = load_field()
    xy_body, n_body, centre, L = load_body()

    print(f"Body centre: {centre},  L={L:.4f}")
    # Rectangle defaults to horizontal (0 deg); rotation_cw_deg tilts it CW
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

    # Hyperparameter optimisation (on sensor observations only)
    mask_v = jnp.ones(len(y_v))
    starts = [np.array([ 0.0,  1.0,  1.0, -4.0]),
              np.array([ 1.0,  0.5,  0.5, -3.0]),
              np.array([ 0.0,  0.0,  0.0, -4.0])]
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

    theta = jnp.array(best.x)

    # Prediction
    xy_te_s = (xy_full - centre) / L
    X_te    = to_4d(jnp.array(xy_te_s))
    pred    = predict(X_tr, y_tr, X_te, theta, mask)
    uv_pred = pred.reshape(-1, 2, order="C") * U_char + [U_inf, V_inf]

    rmse = float(np.sqrt(np.mean((uv_pred - uv_full) ** 2)))
    print(f"n={n_sensors:4d}  SlipWalls={use_wall}  RMSE={rmse:.3f} m/s")

    if plot:
        _plot(xy_full, uv_full, uv_pred, xy_s, xy_body,
              centre, L, theta_deg, n_sensors, use_wall, rmse)
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

    # Build Delaunay triangulation once — reused across all three panels
    tri = Triangulation(xy[:, 0], xy[:, 1])

    # Body polygon (ordered by angle for correct rendering)
    c_b   = xy_body.mean(axis=0)
    order = np.argsort(np.arctan2(xy_body[:, 1] - c_b[1],
                                  xy_body[:, 0] - c_b[0]))
    body_poly = xy_body[order]

    # Sampling region rectangle overlay
    excl_diam = 2.0 * L
    s_r, a_r, h_r = 3.0 * excl_diam, 9.0 * excl_diam, 2.0 * excl_diam
    verts     = _rectangle_vertices(centre, theta_deg, s_r, a_r, h_r)
    rect_poly = np.vstack([verts, verts[0]])

    N_LEVELS = 64   # contour resolution — reduce for more speed, increase for quality

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    panels = [
        (axes[0], speed_t, "Truth |u|",           vmax,            "viridis"),
        (axes[1], speed_p, "GP Recon (Slip) |u|", vmax,            "viridis"),
        (axes[2], err,     "Error Map",            float(err.max()), "magma"),
    ]

    for ax, values, title, vm, cmap in panels:
        cf = ax.tricontourf(tri, values, levels=N_LEVELS,
                            vmin=0, vmax=vm, cmap=cmap)
        # Mask the body interior by drawing a filled white polygon on top
        ax.add_patch(Polygon(body_poly, closed=True,
                             fc="white", ec="k", lw=1.2, zorder=4))
        ax.set_aspect("equal")
        ax.set_title(title)
        fig.colorbar(cf, ax=ax, shrink=0.85)

    # Sensor locations + sampling region on the reconstruction panel
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
    run(num_left=15, num_right=15, num_interior=70,
        seed_string="PEACH_VIBE",
        rotation_cw_deg= -1 * v_flow_deg,
        use_wall=True, plot=True)
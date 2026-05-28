# =============================================================================
# m3_adaptive_back.py
#
# M3  div-free GPR  +  adaptive back-face sampling
# ─────────────────────────────────────────────────
# The downstream back face straddles the wake / free-stream shear layer.
# Start with very few points, iteratively bisect the interval with the
# steepest velocity gradient.
#
# Pressure for the drag integral: scalar 2-D GP (M52) fit on the same
# boundary observations — identical to the original study_grid_3d.py M3.
# "Collocation" in M4/M5 means *continuity-constraint* phantom points;
# M3 never uses them, and neither does this script.
#
# Gradient oracle: interval secant  g_i = |ΔU_speed| / Δs  evaluated
# directly from the GP prediction at the existing sample positions.
# This uses the actual interval width as the FD step — correct scale.
#
# HP re-optimisation every REFIT_EVERY steps so length-scales can shrink
# to resolve the shear layer as back-face density grows.
# =============================================================================

from __future__ import annotations
from pathlib import Path
import hashlib, json, time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import hessian
from scipy.optimize import minimize_scalar, minimize
from scipy.spatial import KDTree, ConvexHull
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

HERE       = Path(__file__).resolve().parent
FIELD_PATH = HERE / "output14_5.xlsx"
CYL_PATH   = HERE / "building14_5.xlsx"

D_TRUTH   = 4207.0
RHO       = 1.225
THETA_DEG = -14.5
N_VAL     = 600

CV_FRONT, CV_BACK, CV_SIDE = 2.0, 4.0, 2.0

N_INIT_BACK  = 5
N_INIT_OTHER = 4
N_MAX_BACK   = 40
GRAD_TOL     = 0.3    # m/s  (velocity jump across interval) in world space
REFIT_EVERY  = 5


# =============================================================================
# §1  Geometry
# =============================================================================

class CVGeom:
    def __init__(self, f, b, s):
        self.front, self.back, self.side = float(f), float(b), float(s)
    def label(self): return f"({self.front},{self.back},±{self.side})D"

def _R(deg):
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])

def _seed(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**32)


# =============================================================================
# §2  Data loading
# =============================================================================

def _clean(df):
    df.columns = [str(c).strip() for c in df.columns]; return df

def _read(p):
    for e in ("openpyxl", "xlrd"):
        try: return _clean(pd.read_excel(p, engine=e))
        except: pass
    raise ValueError(f"Cannot read {p}")

def load_field():
    df = _read(FIELD_PATH)
    xy = df[["x-coordinate", "y-coordinate"]].to_numpy(float)
    uv = df[["x-velocity",   "y-velocity"]].to_numpy(float)
    for c in ["pressure","static-pressure","Pressure","p","P"]:
        if c in df.columns: return xy, uv, df[c].to_numpy(float)
    raise KeyError("pressure column missing")

def _gdiam(xy):
    try:    h = xy[ConvexHull(xy).vertices]
    except: h = xy
    n = len(h); d = 0.; j = 1
    for i in range(n):
        while True:
            nj = (j+1)%n
            if np.sum((h[nj]-h[i])**2) > np.sum((h[j]-h[i])**2): j = nj
            else: break
        d = max(d, np.linalg.norm(h[j]-h[i])); j = (j+1)%n
    return float(d)

def load_body():
    df  = _read(CYL_PATH)
    xy  = np.unique(df[["x-coordinate","y-coordinate"]].to_numpy(float), axis=0)
    cen = xy.mean(0); L = _gdiam(xy)/2.
    return xy, cen, L


# =============================================================================
# §3  Back-face parametrisation
# =============================================================================

def back_face_arc(s_values, cen, L, cv):
    """s ∈ [-cv.side, cv.side] (normalised by L)  →  world (x,y) on back face."""
    R  = _R(THETA_DEG)
    dd = np.array([1.,0.]) @ R.T
    cd = np.array([0.,1.]) @ R.T
    return cen[None,:] + cv.back*L*dd[None,:] + np.asarray(s_values)[:,None]*L*cd[None,:]


# =============================================================================
# §4  Drone placement
# =============================================================================

def place_face(cen, L, cv, n_back, n_other, xy_full):
    """Place drones on all four CV faces.  Returns s_back, bnd_idx."""
    R  = _R(THETA_DEG)
    dd = np.array([1.,0.]) @ R.T
    cd = np.array([0.,1.]) @ R.T
    df, db, ds = cv.front, cv.back, cv.side
    pts, faces = [], []

    s_back = np.linspace(-ds*0.95, ds*0.95, n_back)
    for v in s_back:
        pts.append(cen + db*L*dd + v*L*cd); faces.append(2)
    for v in np.linspace(-ds*0.95, ds*0.95, n_other):
        pts.append(cen - df*L*dd + v*L*cd); faces.append(0)
    for u in np.linspace(-df*0.95, db*0.95, n_other):
        pts.append(cen + u*L*dd - ds*L*cd); faces.append(1)
    for u in np.linspace(-df*0.95, db*0.95, n_other):
        pts.append(cen + u*L*dd + ds*L*cd); faces.append(3)

    _, idx = KDTree(xy_full).query(np.array(pts))
    _, keep = np.unique(idx, return_index=True)
    return s_back, idx[np.sort(keep)]


def snap_fresh(s_new, cen, L, cv, xy_full, current_set):
    """Snap s_new to nearest CFD node not already in current_set."""
    for offset in [0., 0.01, -0.01, 0.03, -0.03, 0.07, -0.07]:
        xy = back_face_arc([s_new + offset], cen, L, cv)
        _, idx = KDTree(xy_full).query(xy)
        fresh = [int(i) for i in np.unique(idx) if i not in current_set]
        if fresh:
            return fresh[0], float(s_new + offset)
    return None, None


# =============================================================================
# §5  M3  div-free GP  (unchanged from study_grid_3d.py)
# =============================================================================

def _m52j(x, xp, var, lx, ly):
    dx = (x[0]-xp[0])/lx; dy = (x[1]-xp[1])/ly
    r  = jnp.sqrt(dx*dx+dy*dy+1e-12); s = jnp.sqrt(5.)*r
    return var*(1.+s+s**2/3.)*jnp.exp(-s)

def _dfe(X, Xp, var, lx, ly):
    x,d   = X[:2], X[2:]
    xp,dp = Xp[:2],Xp[2:]
    H = -hessian(_m52j, argnums=0)(x, xp, var, lx, ly)
    C = jnp.array([[H[1,1],-H[1,0]],[-H[0,1],H[0,0]]])
    return d @ C @ dp

_dfr = jax.vmap(_dfe, in_axes=(None,0,None,None,None))
dfK  = jax.jit(jax.vmap(_dfr, in_axes=(0,None,None,None,None)))

def _to4(xy, uv=None):
    N    = xy.shape[0]
    pos  = jnp.repeat(xy, 2, axis=0)
    dirs = jnp.tile(jnp.eye(2), (N,1))
    X    = jnp.hstack([pos, dirs])
    if uv is not None: return X, uv.reshape(-1, order="C")
    return X

def safe_chol(K, lbl=""):
    n = K.shape[0]; nu = 1e-10
    for _ in range(9):
        try: return np.linalg.cholesky(K + nu*np.eye(n))
        except np.linalg.LinAlgError: nu *= 10
    raise np.linalg.LinAlgError(f"[{lbl}] Chol failed")


def fit_m3(xy_nd, uv_nd, theta_fixed=None):
    JITTER = 1e-8; LX_MIN = np.log(.15)
    X_v, y_v = _to4(jnp.array(xy_nd), jnp.array(uv_nd))
    mv = jnp.ones(len(y_v))
    bds = ((-6,6),(LX_MIN,2.5),(LX_MIN,2.5),(-12,2))
    starts = [np.array([0.,1.,1.,-4.]), np.array([1.,.5,.5,-3.]),
              np.array([-.5,-.5,-.5,-5.]), np.array([.5,2.,.5,-4.]),
              np.array([0.,.3,.3,-4.])]

    if theta_fixed is None:
        def _nll(th):
            th_j = jnp.array(th); vs,lx,ly,noise = jnp.exp(th_j)
            K    = dfK(X_v,X_v,vs,lx,ly) + jnp.diag(noise*mv+JITTER*(1-mv))
            Lc   = jnp.linalg.cholesky(K)
            a    = jax.scipy.linalg.cho_solve((Lc,True),y_v)
            return float(.5*(y_v@a + 2*jnp.sum(jnp.log(jnp.diag(Lc)))
                             + len(y_v)*jnp.log(2*jnp.pi)))
        best = None
        for s0 in starts:
            try:
                r = minimize(_nll, x0=s0, method="L-BFGS-B", bounds=bds)
                if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                    best = r
            except: pass
        if best is None: raise RuntimeError("All M3 optimisation starts failed")
        theta = jnp.array(best.x)
    else:
        theta = jnp.array(theta_fixed)

    vs,lx,ly,noise = jnp.exp(theta)
    K_f  = dfK(X_v,X_v,vs,lx,ly) + jnp.diag(noise*mv+JITTER*(1-mv))
    Lc_f = jnp.linalg.cholesky(K_f)
    alp  = np.array(jax.scipy.linalg.cho_solve((Lc_f,True),y_v))
    return alp, np.array(theta), np.array(X_v)


def pred_m3_uv(alp, theta, X_v, xy_query, cen, L, U_inf, V_inf, U_char, chunk=4000):
    vs,lx,ly,_ = np.exp(theta)
    xy_nd = (xy_query - cen)/L
    Xt    = _to4(jnp.array(xy_nd)); X_vj = jnp.array(X_v)
    pr = []
    for i0 in range(0, len(Xt), chunk):
        pr.append(np.array(dfK(Xt[i0:i0+chunk], X_vj, vs, lx, ly) @ alp))
    uv_nd = np.concatenate(pr).reshape(-1,2,order="C")
    return uv_nd[:,0]*U_char+U_inf, uv_nd[:,1]*U_char+V_inf


# =============================================================================
# §6  Scalar pressure GP  (identical to original M3 in study_grid_3d.py)
# =============================================================================

def _s52(X1, X2, var, lx, ly):
    dx = X1[:,None,0]-X2[None,:,0]; dy = X1[:,None,1]-X2[None,:,1]
    r  = np.sqrt((dx/lx)**2+(dy/ly)**2+1e-12); s = np.sqrt(5.)*r
    return var*(1.+s+s**2/3.)*np.exp(-s)

def fit_sc(Xt, yt, noise=1e-5, hp=None, lbl=""):
    if hp is None: hp = dict(var=float(np.var(yt)+1e-12), lx=1.5, ly=1.0)
    K  = _s52(Xt,Xt,hp["var"],hp["lx"],hp["ly"]) + noise*np.eye(len(Xt))
    Lc = safe_chol(K,lbl); a = np.linalg.solve(Lc.T, np.linalg.solve(Lc,yt))
    return dict(Xt=Xt, a=a, hp=hp)

def pred_sc(m, Xq):
    return _s52(Xq, m["Xt"], m["hp"]["var"], m["hp"]["lx"], m["hp"]["ly"]) @ m["a"]


# =============================================================================
# §7  Drag + RMSE
# =============================================================================

def cv_dense(cen, L, cv, n=600):
    R  = _R(THETA_DEG); dd = np.array([1.,0.])@R.T; cd = np.array([0.,1.])@R.T
    df,db,ds = cv.front,cv.back,cv.side; lf=2*ds; ls=df+db
    ds_=2*(lf+ls)/n; nf=max(2,int(round(lf/ds_))); ns=max(2,int(round(ls/ds_)))
    pts,nr=[],[]
    for v in np.linspace(-ds,ds,nf,endpoint=False):
        pts.append(cen+(-df)*L*dd+v*L*cd); nr.append(-dd)
    for v in np.linspace(-ds,ds,nf,endpoint=False):
        pts.append(cen+ db*L*dd+v*L*cd);   nr.append( dd)
    for u in np.linspace(-df,db,ns,endpoint=False):
        pts.append(cen+u*L*dd+(-ds)*L*cd); nr.append(-cd)
    for u in np.linspace(-df,db,ns,endpoint=False):
        pts.append(cen+u*L*dd+  ds*L*cd);  nr.append( cd)
    return np.array(pts), np.array(nr), 2*(lf+ls)*L/len(pts)

def drag_from(uf, vf, pf, nq, ds, rho):
    R  = _R(THETA_DEG); dd = np.array([1.,0.])@R.T
    uv = np.column_stack([uf,vf]); nd = nq@dd; ud = uv@dd
    un = np.sum(uv*nq, axis=1)
    return float(np.sum((-pf*nd - rho*ud*un)*ds))

def rmse_vel(u_p, v_p, val_idx, uv_full):
    eu = u_p-uv_full[val_idx,0]; ev = v_p-uv_full[val_idx,1]
    return float(np.sqrt(np.mean(eu**2+ev**2)))

def sample_val(xy_full, used_idx, cen, L, n=N_VAL, tag="V"):
    rng   = np.random.default_rng(_seed(tag)); used = set(used_idx.tolist())
    dv    = np.linalg.norm(xy_full-cen, axis=1)
    cands = [i for i in range(len(xy_full))
             if i not in used and 1.2*L <= dv[i] <= 4.*L]
    return rng.choice(cands, size=min(n,len(cands)), replace=False)


def eval_drag_rmse(alp, theta_cached, X_v, bnd_idx,
                    xy_full, uv_full, p_full,
                    cen, L, U_inf, V_inf, U_char, cv, val_idx):
    """Predict drag (M3 velocity + scalar pressure GP) and RMSE."""
    xy_nd = (xy_full[bnd_idx]-cen)/L
    p_nd  = p_full[bnd_idx]/(RHO*U_char**2)
    gp_p  = fit_sc(xy_nd, p_nd, lbl="p")

    xy_q, nq, ds_cv = cv_dense(cen, L, cv)
    u_q,v_q = pred_m3_uv(alp, theta_cached, X_v, xy_q, cen, L, U_inf, V_inf, U_char)
    p_q     = pred_sc(gp_p, (xy_q-cen)/L) * RHO * U_char**2
    D       = drag_from(u_q, v_q, p_q, nq, ds_cv, RHO)
    drag_err = 100.*abs(D-D_TRUTH)/abs(D_TRUTH)

    u_v,v_v = pred_m3_uv(alp, theta_cached, X_v,
                           xy_full[val_idx], cen, L, U_inf, V_inf, U_char)
    rmse = rmse_vel(u_v, v_v, val_idx, uv_full)
    return drag_err, rmse


# =============================================================================
# §8  Refinement criterion  — velocity jump |ΔU| per interval
# =============================================================================

# Minimum interval half-width (in normalised s) below which we stop splitting.
# Prevents infinite subdivision of a single point.
MIN_INTERVAL_WIDTH = 0.05   # ≈ 0.05 * L  in arc-length

def interval_jumps(s_arr, alp, theta, X_v, cen, L, U_inf, V_inf, U_char, cv):
    """
    For each adjacent pair (s[i], s[i+1]) compute the velocity-speed jump:

        j_i = | |U(s[i+1])| - |U(s[i])| |   [m/s]

    This is the total variation of speed across the interval.  Unlike a
    pure secant it does NOT divide by width, so narrow already-refined
    intervals naturally get a small score and the algorithm spreads out
    rather than infinitely subdividing one shear edge.

    Intervals narrower than MIN_INTERVAL_WIDTH are masked to zero so they
    are never selected again.
    """
    xy_pts = back_face_arc(s_arr, cen, L, cv)
    u, v   = pred_m3_uv(alp, theta, X_v, xy_pts, cen, L, U_inf, V_inf, U_char)
    spd    = np.hypot(u, v)

    jumps  = np.abs(np.diff(spd))                       # |ΔU| m/s, shape (n-1,)
    widths = np.abs(np.diff(s_arr))                     # normalised widths

    # zero-out intervals too narrow to bisect further
    jumps[widths < MIN_INTERVAL_WIDTH] = 0.0
    return jumps, widths


# =============================================================================
# §9  Adaptive loop
# =============================================================================

def adaptive_m3(xy_full, uv_full, p_full, cen, L, U_inf, V_inf, U_char, cv):
    s_back_arr, bnd_idx = place_face(cen, L, cv, N_INIT_BACK, N_INIT_OTHER, xy_full)
    s_back = sorted(s_back_arr.tolist())

    val_idx      = sample_val(xy_full, bnd_idx, cen, L)
    history      = []
    theta_cached = None

    print(f"\n{'='*64}")
    print(f"  Adaptive M3  —  CV {cv.label()}")
    print(f"  Initial drones: {len(bnd_idx)}  "
          f"(back={N_INIT_BACK}, per other face={N_INIT_OTHER})")
    print(f"  Budget: {N_MAX_BACK} back-face drones  "
          f"| tol={GRAD_TOL:.1e} m/s (|ΔU| jump)  | refit every {REFIT_EVERY}")
    print(f"{'='*64}\n")

    iteration = 0
    while len(s_back) <= N_MAX_BACK:
        t0 = time.time()

        xy_nd = (xy_full[bnd_idx]-cen)/L
        uv_nd = (uv_full[bnd_idx]-[U_inf,V_inf])/U_char

        need_refit = (theta_cached is None) or (iteration % REFIT_EVERY == 0)
        try:
            alp, theta_cached, X_v = fit_m3(
                xy_nd, uv_nd,
                theta_fixed=(None if need_refit else theta_cached))
            if need_refit:
                vs,lx,ly,noise = np.exp(theta_cached)
                print(f"  [refit iter={iteration}]  "
                      f"lx={lx:.3f} ly={ly:.3f} noise={noise:.2e}")
        except Exception as e:
            print(f"  Iter {iteration}: fit failed — {e}"); break

        try:
            drag_err, rmse = eval_drag_rmse(
                alp, theta_cached, X_v, bnd_idx,
                xy_full, uv_full, p_full,
                cen, L, U_inf, V_inf, U_char, cv, val_idx)
        except Exception as e:
            drag_err = rmse = np.nan
            print(f"  Iter {iteration}: eval failed — {e}")

        elapsed = time.time()-t0
        print(f"  Iter {iteration:3d}  n_back={len(s_back):2d}  "
              f"n_total={len(bnd_idx):3d}  "
              f"drag_err={drag_err:.3f}%  RMSE={rmse:.3f} m/s  ({elapsed:.1f}s)")

        history.append(dict(
            iteration = iteration,
            n_back    = len(s_back),
            n_total   = len(bnd_idx),
            s_back    = s_back.copy(),
            drag_err  = float(drag_err),
            rmse_ms   = float(rmse),
            elapsed_s = elapsed,
        ))

        if len(s_back) >= N_MAX_BACK:
            print("  → budget exhausted."); break

        # ── velocity-jump oracle ─────────────────────────────────────────────
        # Score = |ΔU| across interval (not divided by width).
        # Narrow already-refined intervals get small |ΔU| naturally →
        # algorithm spreads out instead of piling onto one shear edge.
        s_arr          = np.array(s_back)
        jumps, widths  = interval_jumps(s_arr, alp, theta_cached, X_v,
                                         cen, L, U_inf, V_inf, U_char, cv)
        best_i  = int(np.argmax(jumps))
        max_jmp = float(jumps[best_i])

        top3 = np.argsort(jumps)[::-1][:3]
        top3_str = "  ".join(
            f"[{s_arr[i]:.2f},{s_arr[i+1]:.2f}]={jumps[i]:.2f}m/s"
            for i in top3 if jumps[i] > 0)
        print(f"      top jumps: {top3_str}")

        if max_jmp < GRAD_TOL:
            print(f"  → converged (max jump={max_jmp:.2e} < {GRAD_TOL:.2e}), stopping."); break

        # ── bisect ───────────────────────────────────────────────────────────
        s_new = 0.5*(s_arr[best_i] + s_arr[best_i+1])
        node, s_used = snap_fresh(s_new, cen, L, cv, xy_full, set(bnd_idx.tolist()))
        if node is None:
            print("      no fresh node available — stopping."); break

        bnd_idx = np.append(bnd_idx, node)
        s_back.append(s_used); s_back.sort()
        iteration += 1

    print(f"\n  Done.  Final n_back={len(s_back)}, n_total={len(bnd_idx)}")
    return history, bnd_idx, s_back, alp, theta_cached, X_v


# =============================================================================
# §10  Figures
# =============================================================================

def plot_convergence(history, out):
    iters  = [h["iteration"] for h in history]
    n_tot  = [h["n_total"]   for h in history]
    n_back = [h["n_back"]    for h in history]
    d_err  = [h["drag_err"]  for h in history]
    rmse   = [h["rmse_ms"]   for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, xs, xl, tl in [
        (axes[0,0], iters, "Iteration",          "Drag error vs iteration"),
        (axes[0,1], n_tot, "Total drones",        "Drag error vs drone count"),
    ]:
        ax.plot(xs, d_err, "o-", color="#2e6fa3", lw=2)
        for th,col,ls in [(1.,"#00aa00","--"),(2.,"#aaaa00","-."),(5.,"#cc3300",":")]:
            ax.axhline(th, color=col, lw=1.2, ls=ls, label=f"{th}%")
        ax.set_xlabel(xl); ax.set_ylabel("Drag error (%)"); ax.set_title(tl)
        ax.legend(fontsize=8); ax.grid(True,ls=":",alpha=.4); ax.set_yscale("log")

    axes[1,0].plot(iters, rmse, "o-", color="#8b5cf6", lw=2)
    axes[1,0].set_xlabel("Iteration"); axes[1,0].set_ylabel("RMSE [m/s]")
    axes[1,0].set_title("Velocity RMSE vs iteration")
    axes[1,0].grid(True, ls=":", alpha=.4)

    axes[1,1].step(iters, n_back, where="post", color="#4caf7d", lw=2)
    axes[1,1].set_xlabel("Iteration"); axes[1,1].set_ylabel("Back-face drones")
    axes[1,1].set_title("Back-face sample growth")
    axes[1,1].grid(True, ls=":", alpha=.4)

    fig.suptitle("M3 Adaptive Back-Face Sampling — Convergence", fontsize=13)
    fig.tight_layout()
    fig.savefig(out/"fig_convergence.pdf", bbox_inches="tight")
    plt.close(fig); print("  fig_convergence.pdf")


def plot_back_profile(history, cen, L, cv, xy_full, uv_full,
                       alp, theta_m3, X_v, U_inf, V_inf, U_char, out):
    ds = cv.side
    s_dense = np.linspace(-ds*0.98, ds*0.98, 500)
    xy_arc  = back_face_arc(s_dense, cen, L, cv)
    _, idx_arc = KDTree(xy_full).query(xy_arc)
    spd_true = np.hypot(uv_full[idx_arc,0], uv_full[idx_arc,1])

    u_p,v_p = pred_m3_uv(alp, theta_m3, X_v, xy_arc, cen, L, U_inf, V_inf, U_char)
    spd_pred = np.hypot(u_p, v_p)

    n = len(history)-1
    snaps = sorted(set([0]+[min(int(f*n),n) for f in [0.25,0.5,1.0]]))

    fig, axes = plt.subplots(1, len(snaps), figsize=(5*len(snaps),5), sharey=True)
    if len(snaps)==1: axes=[axes]

    for ax, si in zip(axes, snaps):
        h = history[si]; s_b = np.array(h["s_back"])
        ax.plot(s_dense, spd_true, "k-",  lw=1.5, label="CFD truth")
        ax.plot(s_dense, spd_pred, "--",   lw=1.5, color="#2e6fa3", label="M3 (final)")
        for sv in s_b:
            ax.axvline(sv, color="#e07b39", lw=0.8, alpha=0.7)
        ax.set_title(f"Iter {h['iteration']}  n_back={h['n_back']}\n"
                      f"drag={h['drag_err']:.2f}%", fontsize=9)
        ax.set_xlabel("s / L"); ax.grid(True, ls=":", alpha=.4)
    axes[0].set_ylabel("|U| [m/s]"); axes[0].legend(fontsize=7)
    fig.suptitle("Back-face |U| — truth vs M3\nOrange ticks = drones", fontsize=11)
    fig.tight_layout()
    fig.savefig(out/"fig_back_profile.pdf", bbox_inches="tight")
    plt.close(fig); print("  fig_back_profile.pdf")


def plot_secant(history, cen, L, cv, alp, theta_m3, X_v,
                 U_inf, V_inf, U_char, out):
    s_final = np.array(history[-1]["s_back"])
    if len(s_final) < 2: return
    jumps, widths = interval_jumps(s_final, alp, theta_m3, X_v,
                                    cen, L, U_inf, V_inf, U_char, cv)
    s_mids = 0.5*(s_final[:-1]+s_final[1:])

    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(s_mids, jumps, width=widths*0.8,
            color="#2e6fa3", alpha=0.7, label="|ΔU| per interval [m/s]")
    ax.scatter(s_final, np.zeros_like(s_final)-jumps.max()*0.04,
                color="#e07b39", s=60, zorder=5, marker="|",
                linewidths=2, label="Drone positions")
    ax.axhline(GRAD_TOL, color="#cc3300", lw=1.5, ls="--",
                label=f"GRAD_TOL={GRAD_TOL:.1e}")
    ax.set_xlabel("s / L"); ax.set_ylabel("|ΔU| [m/s] per interval")
    ax.set_title("Back-face velocity jump per interval — final\n"
                  "Bars = |ΔU|; orange ticks = drones; grey = min-width mask")
    # shade intervals that are too narrow to split
    for i, w in enumerate(widths):
        if w < MIN_INTERVAL_WIDTH:
            ax.axvspan(s_final[i], s_final[i+1], color="grey", alpha=0.15)
    ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=.4)
    fig.tight_layout()
    fig.savefig(out/"fig_secant.pdf", bbox_inches="tight")
    plt.close(fig); print("  fig_secant.pdf")


def plot_field(xy_full, uv_full, cen, L,
                alp, theta_m3, X_v, U_inf, V_inf, U_char, bnd_idx, out):
    dv   = np.linalg.norm(xy_full-cen, axis=1)
    mask = dv < 5.5*L; xy_r = xy_full[mask]
    spd_t = np.hypot(uv_full[mask,0], uv_full[mask,1])
    u_p,v_p = pred_m3_uv(alp, theta_m3, X_v, xy_r, cen, L, U_inf, V_inf, U_char)
    spd_p = np.hypot(u_p, v_p)
    vmax  = float(np.percentile(spd_t, 98))

    fig, axes = plt.subplots(1,3,figsize=(18,6))
    for ax,title,d,cm,vm in zip(
        axes,
        ["CFD truth |U|","M3 prediction |U|","Error |ΔU|"],
        [spd_t, spd_p, np.abs(spd_p-spd_t)],
        ["viridis","viridis","hot_r"],
        [vmax, vmax, vmax*0.2],
    ):
        sc = ax.scatter(xy_r[:,0],xy_r[:,1],c=d,cmap=cm,vmin=0,vmax=vm,s=2)
        plt.colorbar(sc, ax=ax, shrink=.8)
        ax.scatter(xy_full[bnd_idx,0],xy_full[bnd_idx,1],
                    c="red",s=25,zorder=5,marker="x")
        ax.set_aspect("equal"); ax.set_title(title,fontsize=10)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.suptitle("Field snapshot — M3 adaptive (final)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out/"fig_field.pdf", bbox_inches="tight")
    plt.close(fig); print("  fig_field.pdf")


# =============================================================================
# §11  Main
# =============================================================================

def run():
    print("="*64)
    print(" M3  div-free GP  +  adaptive back-face sampling")
    print("="*64)

    xy_full, uv_full, p_full = load_field()
    _, cen, L = load_body()

    up     = xy_full[:,0] < (cen[0]-3*L)
    U_inf  = float(uv_full[up,0].mean())
    V_inf  = float(uv_full[up,1].mean())
    U_char = float(np.hypot(U_inf, V_inf))
    print(f" L={L:.3f} m   U_char={U_char:.3f} m/s")

    cv  = CVGeom(CV_FRONT, CV_BACK, CV_SIDE)
    out = HERE/"m3_adaptive_out"; out.mkdir(exist_ok=True)

    history, bnd_idx, s_back_final, alp, theta_m3, X_v = adaptive_m3(
        xy_full, uv_full, p_full, cen, L, U_inf, V_inf, U_char, cv)

    with open(out/"adaptive_history.json","w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'Iter':>5}  {'n_back':>7}  {'n_total':>8}  "
          f"{'drag_err(%)':>12}  {'RMSE(m/s)':>10}")
    print("-"*55)
    for h in history:
        print(f"{h['iteration']:5d}  {h['n_back']:7d}  {h['n_total']:8d}  "
              f"{h['drag_err']:12.3f}  {h['rmse_ms']:10.4f}")

    print("\nGenerating figures ...")
    plot_convergence(history, out)
    plot_back_profile(history, cen, L, cv, xy_full, uv_full,
                       alp, theta_m3, X_v, U_inf, V_inf, U_char, out)
    plot_secant(history, cen, L, cv, alp, theta_m3, X_v,
                 U_inf, V_inf, U_char, out)
    plot_field(xy_full, uv_full, cen, L, alp, theta_m3, X_v,
                U_inf, V_inf, U_char, bnd_idx, out)

    print(f"\n Done.  {len(list(out.glob('*.pdf')))} figures → {out}/")
    return history

if __name__ == "__main__":
    run()
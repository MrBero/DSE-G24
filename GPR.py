import os
import pickle

# Force CPU before importing JAX.
os.environ["JAX_PLATFORMS"] = "cpu"

import time
from contextlib import contextmanager

import numpy as np
import scipy.interpolate
import scipy.optimize as spo
import scipy.stats.qmc as qmc
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

import trimesh

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

jax.config.update("jax_enable_x64", True)

from flowpanelwrapper import FLOWPanelSolver
from sampling import sample

from alive_progress import alive_bar
import time
import builtins

#this might break prints so comment if bad
_original_print = builtins.print
def _print(*args, **kwargs):
    _original_print(f"[{time.perf_counter():.3f}]   ", *args, **kwargs)

builtins.print = _print

# Kernelissimo Kernelismus
def matern52_np(v1, v2, ell, var):
    """Scalar Matérn-5/2 covariance. v1,v2:(3,)  ell:(3,)  var:scalar."""
    diff = v2 - v1
    r = jnp.sqrt(
        (diff[0] / ell[0]) ** 2
        + (diff[1] / ell[1]) ** 2
        + (diff[2] / ell[2]) ** 2
        + 1e-8 # so we don't run into strange stuff at r=0
    )
    return var * (1.0 + jnp.sqrt(5.0) * r + (5.0 / 3.0) * r ** 2) * jnp.exp(-jnp.sqrt(5.0) * r)

# full derivation of 3x3 k0 matrix in the final report
#compile just in time for speed
@jax.jit
def Hemholtz_K0(V1, V2, ell, var):
    """3x3 divergence-free vector kernel block from Hessian of scalar Matérn-5/2."""
    H = jax.hessian(matern52_np)(V1, V2, ell, var)
    return jnp.array(
        [
            [-H[1, 1] - H[2, 2], H[0, 1], H[0, 2]],
            [H[1, 0], -H[2, 2] - H[0, 0], H[1, 2]],
            [H[2, 0], H[2, 1], -H[0, 0] - H[1, 1]],
        ]
    )
# assemble the covariance matrix from K0 matrices 
def assemble_dat_shi(points_1, points_2, ell, var, noise_std=0.0, jitter=0.0):
    """
    Block vector-valued covariance, point-major ordering:
        [u_x(p0), u_y(p0), u_z(p0), u_x(p1), ...]
    Returns (3*len(points_1), 3*len(points_2)).
    """
    points_1 = jnp.asarray(points_1)
    points_2 = jnp.asarray(points_2)
    n_1 = points_1.shape[0]
    n_2 = points_2.shape[0]

    blocks = jax.vmap(
        lambda a: jax.vmap(lambda b: Hemholtz_K0(a, b, ell, var))(points_2)
    )(points_1)

    result_matrix = jnp.transpose(blocks, (0, 2, 1, 3)).reshape(n_1 * 3, n_2 * 3)

    if n_1 == n_2:
        if noise_std > 0.0:
            result_matrix = result_matrix + (noise_std ** 2 + jitter) * jnp.eye(n_1 * 3)
        elif jitter > 0.0:
            result_matrix = result_matrix + jitter * jnp.eye(n_1 * 3)

    return result_matrix

# Hyperparam fitting, the fun stuff
def fit_hyperparams(train_coords, train_residuals, n_restarts=8, jitter=1e-6, seed=0):
    
    X = jnp.asarray(train_coords) #training coordinates list, loaded into jax numpy
    y = jnp.asarray(train_residuals).reshape(-1, 1) #load y 
    n = X.shape[0] # number of points
    #lower and upper bound in log space 
    lo = np.log([5.0,  5.0,  5.0,  1e-1, 1e-2])
    hi = np.log([150.0, 150.0, 150.0, 1e3,  2e0])

    @jax.jit
    #find negative log likelihood
    def nll(log_theta):
        ell = jnp.exp(log_theta[:3]) # exponentiate length scales to convert back from log space
        var = jnp.exp(log_theta[3]) # same with variance
        noise = jnp.exp(log_theta[4]) # same with noise
        #assemble K matrix (as above)
        blocks = jax.vmap(lambda a: jax.vmap(lambda b: Hemholtz_K0(a, b, ell, var))(X))(X)
        K = jnp.transpose(blocks, (0, 2, 1, 3)).reshape(3 * n, 3 * n)
        K += (noise ** 2 + jitter) * jnp.eye(3 * n) #add noise and jitter on the diagnonal 
        
        c, low = cho_factor(K) #apply cholesky decomp (K = LL^T)
        alpha = cho_solve((c, low), y) #solve for weights
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(c))) # calculate the determinant by using the fact that it's the sum of diagonal entries of c squared. 
        #return negative log likelihood, this is pretty standard
        return 0.5 * (y.T @ alpha)[0, 0] + 0.5 * logdet + 0.5 * (3 * n) * jnp.log(2 * jnp.pi) 
    #find value and gradient using autodiff for the objective function
    nll_vg = jax.jit(jax.value_and_grad(nll))
    #objective function, helper for the spo optimizer later
    def objective(log_theta):
        val, grad = nll_vg(jnp.asarray(log_theta))
        return float(val), np.asarray(grad, dtype=float)

    #starting points spaced in the lo-high space in a latin hypercube to cover the space
    starts = lo + qmc.LatinHypercube(d=5, seed=seed).random(n_restarts) * (hi - lo)

    #optimizer loop 
    best = None
    for t0 in starts:
        try:
            # optimizer function for spo using the L-BFGS-B method, using autodiff from objective, between higher and lower bound, max 200 steps.
            res = spo.minimize(
                objective, t0, method="L-BFGS-B", jac=True,
                bounds=list(zip(lo, hi)), options={"maxiter": 200}
            )
            if best is None or res.fun < best.fun:
                best = res
        except Exception as e:
            print(f"Restart failed")

    if best is None:
        raise RuntimeError("All optimizer restarts crashed. The math broke.")
    #recover theta from log space
    theta = np.exp(best.x)
    #return hyperparams
    return {"ell": theta[:3], "var": float(theta[3]), "noise": float(theta[4]),
            "sample_spacing": 0.0, "nll": float(best.fun)}

# posterior mean calculation, originally computed as one. We ran into memory issues... so now it is batched in sets of =batch points.
def posterior_mean_batched(test_points, training_coords, ell, var, alpha, means_tests,
                           batch=4000, progress_every=0):
    """Stream the GP posterior mean over chunks of test points. Returns (3*n_test, 1)."""
    #get test points, ie the points that we evaluate the function at 
    test_points = np.asarray(test_points)
    means_tests = np.asarray(means_tests).reshape(-1, 3)
    n_test = test_points.shape[0]
    out = np.empty((n_test, 3), dtype=float) 
    alpha_local = jnp.asarray(alpha) # reminder: alpha is (K + sigma^2)^-1 (y-y_mean)
    n_chunks = (n_test + batch - 1) // batch
    
    for ci, i in enumerate(range(0, n_test, batch)):
        tp = test_points[i:i + batch] 
        # cross-covariance K(X_*, X)
        ks = assemble_dat_shi(tp, training_coords, ell, var, noise_std=0.0, jitter=0.0)
        # now do 
        contrib = np.array(ks @ alpha_local).reshape(-1, 3) # calculate prediction at the point by doing k(x_*, x) @ (K + sigma^2)^-1 (y-y_mean)
        out[i:i + batch] = means_tests[i:i + batch] + contrib # write output by adding mean to deviation prediction
        if progress_every and (ci % progress_every == 0 or ci == n_chunks - 1):
            print(f"    posterior vels chunk {ci + 1}/{n_chunks}", flush=True)

    return out.reshape(-1, 1)

# calculate the posterior variance in batches... again, because the matrix was taking up 23 gb of ram. 
# EThe equation we're using here for variance is: Sigma_* = K(X_*, X_*) - K(X_*, X) * (K(X, X) + noise^2 I)^-1 * K(X_*, X)^T
def posterior_vars_batched(test_points, training_coords, ell, var, c, low,
                           batch=2000, progress_every=0):

    test_points = np.asarray(test_points)
    n_test = test_points.shape[0]
    out = np.empty((n_test * 3,), dtype=float)
    n_chunks = (n_test + batch - 1) // batch

    for ci, i in enumerate(range(0, n_test, batch)):
        chunk = test_points[i:i + batch]
        m = chunk.shape[0]
        
        K_tt = assemble_dat_shi(chunk, chunk, ell, var)              # (3m, 3m)
        # 
        k_tc = assemble_dat_shi(chunk, training_coords, ell, var)    # (3m, 3n)
        beta_chunk = cho_solve((c, low), jnp.asarray(k_tc.T))        # (3n, 3m)
        diag = jnp.diag(K_tt - k_tc @ beta_chunk)                    # (3m,)

        out[i * 3:(i + m) * 3] = np.asarray(diag)
        if progress_every and (ci % progress_every == 0 or ci == n_chunks - 1):
            print(f"    posterior vars chunk {ci + 1}/{n_chunks}", flush=True)
    return out


# helpers

def rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def velocity_magnitude(U):
    return np.sqrt(np.sum(U ** 2, axis=-1))


def predict_batched(estimator, test_points, batch=4000):
    """Stream estimator.predict over chunks to bound peak memory."""
    test_points = np.asarray(test_points)
    n_test = test_points.shape[0]
    out = np.empty(n_test, dtype=float)
    for i in range(0, n_test, batch):
        out[i:i + batch] = estimator.predict(test_points[i:i + batch])
    return out

def ts():
    return time.perf_counter()


# =============================================================================
# Momentum-integral force (separate pass - never touches the grid arrays)
# =============================================================================

def _show_momentum_plot(mesh, pcd):
    """Optional pyvista view of the momentum surface (his original visualization)."""
    try:
        import pyvista
    except Exception:
        print("pyvista not available - skipping momentum plot", flush=True)
        return
    plotter = pyvista.Plotter()
    plotter.add_mesh(mesh, scalars="pressure")
    arrows = mesh.glyph(orient="Normals", scale=False, factor=5)
    plotter.add_mesh(arrows, color="red")
    plotter.show()
    plotter.close()

def compute_momentum_force(cylinder_geom, solver, v_inf,
                           training_coords, ell, var, alpha, means_tests_dummy,
                           p_gpr, posterior_batch,
                           n_points=25000, include_top=True, include_bottom=False,
                           bpa_L=10.0, midpoint=None, show_plot=False,
                           prior_cache_dir="prior_cache"):
    """Evaluate the momentum-balance surface force on the cylinder control volume.

    This is a SELF-CONTAINED pass: it generates the momentum surface points, gets
    their prior (panel solver, cached), GP velocity posterior, and pressure
    posterior, then computes the surface force. It never appends to or reads the
    res^3 grid, so plots and grid RMSE are unaffected.

    The surface-force step (pyvista/VTK + open3d BPA) runs in a SPAWNED child
    process. VTK keeps process-lifetime global singletons that leak into the
    parent's terminal/stdout state and break the alive_bar on subsequent runs;
    isolating it in a child that exits cleanly is the only reliable teardown.
    Because of this, "mesh" and "pcd" are returned as None (they live in the
    now-dead child); nothing downstream consumes them when show_plot is False.

    Returns dict: {"force", "mesh", "pcd", "points", "velocity", "pressure", "n"}.
    """
    from bektir_experimentatation.epstein3 import make_oblique_cylinder_mesh, surface_force as epstein_surface_force

    R = float(cylinder_geom["R"])
    p1 = np.asarray(cylinder_geom["bottom_center"], float)
    p2 = np.asarray(cylinder_geom["top_center"], float)

    # --- momentum surface mesh (triangulated mesh via Epstein3) ---
    mesh = make_oblique_cylinder_mesh(
        center_bot=p1, center_top=p2, radius=R,
        total_points=n_points, cap_bottom=include_bottom, cap_top=include_top
    )
    region_points = np.asarray(mesh.points, float)
    n_mom = region_points.shape[0]

    # --- prior on the surface (panel solver), cached by geometry ---
    os.makedirs(prior_cache_dir, exist_ok=True)
    geo_key = (f"epstein3_R{R:g}_b{p1[0]:g}_{p1[1]:g}_{p1[2]:g}"
               f"_t{p2[0]:g}_{p2[1]:g}_{p2[2]:g}_n{n_mom}"
               f"_top{int(include_top)}_bot{int(include_bottom)}"
               f"_vinf{v_inf[0]:g}_{v_inf[1]:g}_{v_inf[2]:g}")
    mom_prior_file = os.path.join(prior_cache_dir, f"momentum_prior_{geo_key}.pkl")
    if os.path.exists(mom_prior_file):
        with open(mom_prior_file, "rb") as f:
            mom_prior = pickle.load(f)
        if mom_prior.shape != (n_mom, 3):
            raise RuntimeError(
                f"Cached momentum prior {mom_prior.shape} != {(n_mom, 3)}; "
                f"delete {mom_prior_file} and rerun.")
        print(f"loaded cached momentum-surface prior ({n_mom} pts)", flush=True)
    else:
        print(f"computing momentum-surface prior via Julia ({n_mom} pts)...", flush=True)
        mom_prior = np.empty((n_mom, 3), dtype=float)
        for i in range(0, n_mom, posterior_batch):
            chunk = region_points[i:i + posterior_batch]
            mom_prior[i:i + posterior_batch] = solver.velocity(
                chunk, blank_interior=True).reshape(-1, 3)
        with open(mom_prior_file, "wb") as f:
            pickle.dump(mom_prior, f)
        print(f"saved momentum-surface prior cache", flush=True)

    # --- GP velocity posterior on the surface (separate batched pass) ---
    mom_vel = posterior_mean_batched(
        region_points, training_coords, ell, var, alpha, mom_prior,
        batch=posterior_batch, progress_every=0)
    mom_vel = np.asarray(mom_vel).reshape(-1, 3)

    # --- pressure posterior on the surface ---
    mom_p = predict_batched(p_gpr, region_points, batch=posterior_batch) \
        if p_gpr is not None else np.zeros(n_mom)

    # --- surface force via momentum balance using Epstein3 quadratic integration ---
    mesh.point_data["velocity"] = mom_vel
    mesh.point_data["pressure"] = mom_p
    
    # epstein3 logic uses analytical integration on the triangulated mesh
    FORCE, result_mesh = epstein_surface_force(mesh, rho=1.225)

    if show_plot:
        _show_momentum_plot(result_mesh, None)

    return {"force": np.asarray(FORCE, float), "mesh": result_mesh, "pcd": None,
            "points": region_points, "velocity": mom_vel, "pressure": mom_p,
            "n": n_mom}

# =============================================================================
# Sampling defaults per method
# =============================================================================

# Each sampling method takes its own config keys. When main() doesn't pass a
# sample_config, fall back to the method-appropriate default below.
SAMPLE_DEFAULTS = {
    "cylinder": {"r_factor": 1.5, "h_factor": 2.0, "tilt_deg": 30, "n_points": 300},
    "drone_array": {"tilt_deg": 30, "n_rows": 10, "n_cols": 10},
    "random": {},
    "CSV": {},
    "array": {},
}

# =============================================================================
# Callable pipeline
# =============================================================================

def run_gpr(
    stl_filepath="input_stls/triangle.stl",
    cfd_filepath="inputs/FLTG.csv",
    stl_scale=1.0 / 1000.0,
    stl_rotate = None,
    bounds_input = None,
    res=150,
    posterior_batch=4000,
    v_inf=(12.0, 0.0, 0.0),
    n_restarts=6,
    fit_pressure=True,
    sample_method="cylinder",
    sample_config=None,
    samples=None,
    num_samples=150,
    compute_variance=True,
    var_res=None,
    cylinder_geom_override=None,
    compute_force=True,
    momentum_n_points=25000,
    momentum_include_top=True,
    momentum_include_bottom=False,
    momentum_bpa_L=10.0,
    momentum_midpoint=None,
    momentum_show_plot=False):

    v_inf = np.asarray(v_inf, dtype=float)

    n_test_est = res ** 3
    print(f"\nrun_gpr: res={res} -> {n_test_est:,} test points, "
            f"batch={posterior_batch}, n_restarts={n_restarts}, "
            f"fit_pressure={fit_pressure}\n", flush=True)

    with alive_bar(8, dual_line=True, enrich_print=False) as bar:
        bar.text('Loading...')
        # --- Mesh ---
        stl_mesh = trimesh.load_mesh(stl_filepath)
        if stl_scale != 1.0:
            stl_mesh.apply_scale(stl_scale)
        if stl_rotate is not None:
            centroid = stl_mesh.centroid
            rot_matrix = trimesh.transformations.rotation_matrix(angle=stl_rotate, direction=[0,0,1], point=centroid)
            stl_mesh.apply_transform(rot_matrix)

        # --- Panel solver (prior) ---
        solver = FLOWPanelSolver(stl_mesh, v_inf, julia_script="FP.jl",
                                julia_bin="julia", verbose=False)
        bar()

        bar.text('Sampling...')
        # --- Sample training data ---
        ground_truth, bounds, sample_dat_shi, cylinder_geom = sample(                           #BIKTOR HERE IS GEOMETRY
        cfd_filepath, stl_mesh, sample_method=sample_method, num_samples=num_samples,
        sample_config=sample_config, samples=samples, v_inf=v_inf,
        epsilon=0.02, use_signed_distance=True,)
    
        if bounds_input is not None:
            bounds = bounds_input

        bar.text('Training points...')
        training_coords = ground_truth[["x-target", "y-target", "z-target"]].to_numpy()
        training_vels = ground_truth[["x-velocity", "y-velocity", "z-velocity"]].to_numpy().reshape(-1, 1)
        training_point_n = len(ground_truth)
        if training_point_n == 0:
            raise RuntimeError("No training points were sampled.")
        print(f"training points: {training_point_n}")

        bar.text('Test points...')
        # --- Test grid ---
        gx = np.linspace(bounds[0, 0], bounds[0, 1], res)
        gy = np.linspace(bounds[1, 0], bounds[1, 1], res)
        gz = np.linspace(bounds[2, 0], bounds[2, 1], res)
        x, y, z = np.meshgrid(gx, gy, gz, indexing="ij")
        test_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
        del x, y, z
        print(f'testing points: {test_points.shape[0]}')
        bar()

        bar.text('Computing prior means (potential flow solver in Julia)...')
        # --- Prior mean (panel solver), streamed over the grid ---
        n_test = test_points.shape[0]
        n_chunks = (n_test + posterior_batch - 1) // posterior_batch

        # Cache the Julia prior on the grid. 
        # Delete prior_cache if CFD/Geometry/Tolerances change.
        os.makedirs("prior_cache", exist_ok=True)
        prior_file = os.path.join(
            "prior_cache",
            f"grid_prior_res{res}_vinf{v_inf[0]:g}_{v_inf[1]:g}_{v_inf[2]:g}.pkl",
        )

        if os.path.exists(prior_file):
            with open(prior_file, "rb") as f:
                means_tests = pickle.load(f)
            if means_tests.shape != (n_test, 3):
                raise RuntimeError(
                    f"Cached prior {means_tests.shape} != expected {(n_test, 3)}. "
                    f"Delete {prior_file} and rerun."
                )
            print(f"loaded cached grid prior from {prior_file} (skipping Julia)", flush=True)
        else:
            tolerance = True
            if tolerance:
                near_tol = 0.04 * solver.diag
                print(f"near_tol = {near_tol:.4g} (median panel edge)", flush=True)
            means_tests = np.empty((n_test, 3), dtype=float)
            print("contains center:", solver.mesh.contains(solver.mesh.bounds.mean(axis=0).reshape(1, 3)),
                  " is_watertight:", solver.mesh.is_watertight,
                  "is_winding_consistent:", solver.mesh.is_winding_consistent)
            
            for ci, i in enumerate(range(0, n_test, posterior_batch)):
                chunk = test_points[i:i + posterior_batch]
                if tolerance:
                    means_tests[i:i+posterior_batch] = solver.velocity(chunk, blank_interior=True, blank_near=True, near_tol=near_tol).reshape(-1, 3)
                else:
                    means_tests[i:i+posterior_batch] = solver.velocity(chunk, blank_interior=True).reshape(-1, 3)
                if (ci % 50 == 0 or ci == n_chunks - 1):
                    print(f"prior chunk {ci + 1}/{n_chunks}", flush=True)
            with open(prior_file, "wb") as f:
                pickle.dump(means_tests, f)
            print(f"saved grid prior to {prior_file}", flush=True)

        
        means_training = solver.velocity(training_coords, blank_interior=False).reshape(-1, 3)
        if np.isnan(means_training).any():
            raise RuntimeError("NaNs in direct prior mean at training points.")
        means_training = means_training.reshape(-1, 1)

        prior_train_rmse = rmse(means_training, training_vels)
        bar()

        bar.text('Fit hyperparameters')
        # --- Fit GP to residuals ---
        residuals = training_vels - means_training
        # print(training_vels)

        if np.isnan(residuals).any():
            raise RuntimeError("NaNs in residuals_for_fit.")

        fit = fit_hyperparams(training_coords, residuals, n_restarts=n_restarts, jitter=1e-4, seed=0)
        
        print(f"    nll={fit['nll']:.6g}  ell={fit['ell']}  "
            f"var={fit['var']:.4g}  noise={fit['noise']:.4g}", flush=True)

        ell = jnp.asarray(fit["ell"])
        var = float(fit["var"])
        noise = float(fit["noise"])
        bar()

        bar.text('Assemble K + cholesky solve (invert K) at residuals...')
        # --- Solve for alpha, build posterior ---
        K = assemble_dat_shi(training_coords, training_coords, ell, var, noise_std=noise, jitter=1e-8)
        c, low = cho_factor(K)
        alpha = cho_solve((c, low), jnp.asarray(residuals)) #inverted K matrix times residuals is alpha
        bar()

        bar.text(f"Velocity posterior over grid ({n_chunks} chunks)...")
        GPR_posterior = posterior_mean_batched(
            test_points, training_coords, ell, var, alpha, means_tests,
            batch=posterior_batch, progress_every=posterior_batch*10)
        GPR_posterior = np.array(GPR_posterior).reshape(-1, 3)

        bar.text("Variance posterior...")
        if compute_variance:
            if var_res is not None and var_res < res:
                # --- coarse variance pass + trilinear interpolation up to res^3 ---
                # Variance is the slow term, so evaluate it on a var_res^3 grid and
                # interpolate back onto the full res^3 grid. PLOT.py reshapes every
                # field with the single `res`, so the returned array MUST be res^3:
                # interpolating up keeps that contract intact.
                cgx = np.linspace(bounds[0, 0], bounds[0, 1], var_res)
                cgy = np.linspace(bounds[1, 0], bounds[1, 1], var_res)
                cgz = np.linspace(bounds[2, 0], bounds[2, 1], var_res)
                cx, cy, cz = np.meshgrid(cgx, cgy, cgz, indexing="ij")
                coarse_pts = np.stack([cx.ravel(), cy.ravel(), cz.ravel()], axis=-1)
                print(f"variance: coarse grid {var_res}^3 = {coarse_pts.shape[0]:,} "
                      f"pts (vs {res**3:,} full)", flush=True)

                coarse_var = posterior_vars_batched(
                    coarse_pts, training_coords, ell, var, c, low,
                    batch=posterior_batch, progress_every=posterior_batch*10)
                coarse_var = np.array(coarse_var).reshape(var_res, var_res, var_res, 3)

                from scipy.interpolate import RegularGridInterpolator
                interp = RegularGridInterpolator(
                    (cgx, cgy, cgz), coarse_var,
                    bounds_error=False, fill_value=None)  # None -> extrapolate edges
                GPR_variances = interp(test_points).reshape(-1, 3)
            else:
                GPR_variances = posterior_vars_batched(
                    test_points, training_coords, ell, var, c, low,
                    batch=posterior_batch, progress_every=posterior_batch*10)
                GPR_variances = np.array(GPR_variances).reshape(-1, 3)
        else:
            GPR_variances = np.full((test_points.shape[0], 3), np.nan)

        # Training reconstruction check
        K_signal = assemble_dat_shi(training_coords, training_coords, ell, var, noise_std=0.0, jitter=0.0)
        train_post = jnp.asarray(means_training) + K_signal @ alpha
        posterior_train_rmse = rmse(np.array(train_post), training_vels)
        bar()

        bar.text('Interpolating truth for comparison...')
        # --- CFD truth on test grid (velocity + pressure in one griddata pass) ---
        
        cfd_test = sample_dat_shi(test_points)

        cfd_test_vels = cfd_test[:, :3]
        cfd_p = cfd_test[:, 3]

        # A point is usable for velocity RMSE only if the CFD truth AND both
        # predicted fields are finite there. Masking only the CFD truth (the old
        # behavior) let a single NaN/Inf in the prior or posterior poison the
        # whole metric -> post_test_rmse=nan.
        valid_cfd = (
            ~np.any(np.isnan(cfd_test_vels), axis=1)
            & np.all(np.isfinite(means_tests), axis=1)
            & np.all(np.isfinite(GPR_posterior), axis=1)
        )
        n_bad_pred = int((~np.all(np.isfinite(GPR_posterior), axis=1)).sum())
        if n_bad_pred:
            print(f"WARNING: {n_bad_pred} non-finite posterior points excluded "
                  f"from RMSE (check var/noise bounds).", flush=True) #flags building inside
        truth = cfd_test_vels[valid_cfd]
        prior_test_rmse = rmse(means_tests[valid_cfd], truth)
        post_test_rmse = rmse(GPR_posterior[valid_cfd], truth)
        truth_test_rms = float(np.sqrt(np.mean(truth ** 2)))

        bar.text('Pressure GPR...')
        # --- Pressure GPR (scalar, sklearn) ---
        pressure_posterior = None
        pressure_test_rmse = None
        if fit_pressure:
            train_p = ground_truth["pressure"].to_numpy()
            p_kernel = Matern(
                length_scale=[1.0, 1.0, 1.0], 
                length_scale_bounds=(1e-2, 1e3), 
                nu=2.5)
                
            p_gpr = GaussianProcessRegressor(
                kernel=p_kernel, normalize_y=True, n_restarts_optimizer=8, random_state=0,
            ).fit(training_coords, train_p)
            print(f"fitted kernel: {p_gpr.kernel_}", flush=True)

            pressure_posterior = predict_batched(p_gpr, test_points, batch=posterior_batch)

            valid_p = ~np.isnan(cfd_p)
            pressure_test_rmse = rmse(pressure_posterior[valid_p], cfd_p[valid_p])
        bar()

        bar.text('Force Calculation...')
        # --- Momentum-integral surface force (SEPARATE pass, never touches grid) ---
        momentum_result = None
        force_vec = None
        force_mag = None
        geom_for_force = cylinder_geom if cylinder_geom is not None else cylinder_geom_override
        if compute_force and geom_for_force is not None:
            try:
                p_gpr_local = p_gpr if fit_pressure else None
                momentum_result = compute_momentum_force(
                    geom_for_force, solver, v_inf,
                    training_coords, ell, var, alpha, None,
                    p_gpr_local, posterior_batch,
                    n_points=momentum_n_points,
                    include_top=momentum_include_top,
                    include_bottom=momentum_include_bottom,
                    bpa_L=momentum_bpa_L, midpoint=momentum_midpoint,
                    show_plot=momentum_show_plot)
                force_vec = momentum_result["force"]
                force_mag = float(np.linalg.norm(force_vec))
                print(f"momentum force: |F|={force_mag:.6g}  F={np.asarray(force_vec)}",
                      flush=True)
            except Exception as e:
                print(f"WARNING: momentum force failed ({e}); continuing without it.",
                      flush=True)
        bar()

        metrics = {
            "training_point_n": training_point_n,
            "prior_train_rmse": prior_train_rmse,
            "posterior_train_rmse": posterior_train_rmse,
            "prior_test_rmse": prior_test_rmse,
            "post_test_rmse": post_test_rmse,
            "truth_test_rms": truth_test_rms,
            "rel_prior_test_rmse": prior_test_rmse / max(truth_test_rms, 1e-12),
            "rel_post_test_rmse": post_test_rmse / max(truth_test_rms, 1e-12),
            "valid_cfd": int(valid_cfd.sum()),
            "n_test": int(len(valid_cfd)),
            "pressure_test_rmse": pressure_test_rmse,
            "force_vec": (force_vec.tolist() if force_vec is not None else None),
            "force_mag": force_mag,
            "momentum_n": (momentum_result["n"] if momentum_result is not None else 0),
        }
        bar.text('Done!')
    

    _result = {
        "test_points": test_points,
        "bounds": bounds,
        "res": res,
        "means_tests": means_tests,
        "GPR_posterior": GPR_posterior,
        "GPR_variances": GPR_variances,
        "cfd_test_vels": cfd_test_vels,
        "pressure_posterior": pressure_posterior,
        "training_coords": training_coords,
        "mesh_vertices": np.asarray(solver.mesh.vertices),
        "fit": fit,
        "metrics": metrics,
        'V_inf': v_inf,
        "cylinder_geom": cylinder_geom,
        "momentum": momentum_result,   # {force, mesh, pcd, points, velocity, pressure, n} or None
        # --- additive: exposed for adaptive sampling (cheap local re-evaluation) ---
        "ell": np.asarray(ell, dtype=float),
        "var": float(var),
        "noise": float(noise),
        "alpha": np.asarray(alpha, dtype=float),
        "sample_dat_shi": sample_dat_shi,   # live CFD sampler closure
        "stl_mesh": stl_mesh,               # for mesh-reject of proposed points
    }

    # Shut down THIS run's Julia panel server now, rather than waiting for the
    # atexit hook. Each run_gpr starts a fresh FLOWPanelSolver -> a fresh Julia
    # process; if they're only closed at program exit they accumulate across
    # phases/configs and exhaust memory. mesh_vertices is already a numpy copy,
    # so nothing in the returned dict needs the live solver.
    try:
        solver.close()
    except Exception:
        pass

    return _result


if __name__ == "__main__":
    result = run_gpr(verbose=False)
    print("\nMetrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")
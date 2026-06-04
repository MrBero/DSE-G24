import os

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

from potential_flow_run import *
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
        + 1e-8
    )
    return var * (1.0 + jnp.sqrt(5.0) * r + (5.0 / 3.0) * r ** 2) * jnp.exp(-jnp.sqrt(5.0) * r)

# full derivation of 3x3 k0 matrix in the final report
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
    
    X = jnp.asarray(train_coords)
    y = jnp.asarray(train_residuals).reshape(-1, 1)
    n = X.shape[0]

    lo = np.full(5, np.log(1e-5))
    hi = np.full(5, np.log(1e5))

    @jax.jit
    def nll(log_theta):
        ell = jnp.exp(log_theta[:3])
        var = jnp.exp(log_theta[3])
        noise = jnp.exp(log_theta[4])
        
        blocks = jax.vmap(lambda a: jax.vmap(lambda b: Hemholtz_K0(a, b, ell, var))(X))(X)
        K = jnp.transpose(blocks, (0, 2, 1, 3)).reshape(3 * n, 3 * n)
        K += (noise ** 2 + jitter) * jnp.eye(3 * n)
        
        c, low = cho_factor(K)
        alpha = cho_solve((c, low), y)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(c)))
        
        return 0.5 * (y.T @ alpha)[0, 0] + 0.5 * logdet + 0.5 * (3 * n) * jnp.log(2 * jnp.pi)

    nll_vg = jax.jit(jax.value_and_grad(nll))

    def objective(log_theta):
        val, grad = nll_vg(jnp.asarray(log_theta))
        return float(val), np.asarray(grad, dtype=float)

    # 2. Latin Hypercube is maintained, but now stretches across the massive generic box
    starts = lo + qmc.LatinHypercube(d=5, seed=seed).random(n_restarts) * (hi - lo)

    best = None
    for t0 in starts:
        try:
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

    theta = np.exp(best.x)
    
    return {"ell": theta[:3], "var": float(theta[3]), "noise": float(theta[4]),
            "sample_spacing": 0.0, "nll": float(best.fun)}


def posterior_mean_batched(test_points, training_coords, ell, var, alpha, means_tests,
                           batch=4000, progress_every=0):
    """Stream the GP posterior mean over chunks of test points. Returns (3*n_test, 1)."""
    test_points = np.asarray(test_points)
    means_tests = np.asarray(means_tests).reshape(-1, 3)
    n_test = test_points.shape[0]
    out = np.empty((n_test, 3), dtype=float)
    alpha_local = jnp.asarray(alpha)
    n_chunks = (n_test + batch - 1) // batch

    for ci, i in enumerate(range(0, n_test, batch)):
        tp = test_points[i:i + batch]
        ks = assemble_dat_shi(tp, training_coords, ell, var, noise_std=0.0, jitter=0.0)
        contrib = np.array(ks @ alpha_local).reshape(-1, 3)
        out[i:i + batch] = means_tests[i:i + batch] + contrib
        if progress_every and (ci % progress_every == 0 or ci == n_chunks - 1):
            print(f"    posterior chunk {ci + 1}/{n_chunks}", flush=True)

    return out.reshape(-1, 1)

def posterior_vars_batched(test_points, training_coords, ell, var, beta,
                           batch=300, progress_every=0):
    test_points = np.asarray(test_points)
    n_test = test_points.shape[0]
    out = np.empty((n_test*3), dtype=float)
    beta_local = jnp.asarray(beta)
    n_chunks = (n_test + batch - 1) // batch

    for ci, i in enumerate(range(0, n_test, batch)):
        chunk = test_points[i:i+batch]
        n_chunk = chunk.shape[0]

        K_tests_slice = jnp.array(assemble_dat_shi(chunk, chunk, ell, var))
        k_slice = jnp.array(assemble_dat_shi(chunk, training_coords, ell, var))
        
        beta_slice = np.array(beta[:, i*3:(i+batch)*3])
        contrib = k_slice @ beta_slice
        out[i*3:(i + batch)*3] = jnp.diag(K_tests_slice - contrib)
        if progress_every and (ci % progress_every == 0 or ci == n_chunks - 1):
            print(f"    posterior chunk {ci + 1}/{n_chunks}", flush=True)
    print(out)
    return out


# =============================================================================
# Small helpers
# =============================================================================

def rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def velocity_magnitude(U):
    return np.sqrt(np.sum(U ** 2, axis=-1))


def interpolate_cfd_to_points(cfd_filepath, query_points, columns, batch=200_000):
    """
    Linear-interpolate CFD columns onto query points.

    Builds the Delaunay triangulation ONCE (over the CFD source cloud) via
    LinearNDInterpolator, then evaluates the query in chunks so the query-side
    allocations stay bounded. The triangulation itself is sized by the source
    cloud, not the query, and is unavoidable for linear scattered interpolation.
    """
    df = pd.read_csv(cfd_filepath)
    df.columns = df.columns.str.strip()
    coords = df[["x-coordinate", "y-coordinate", "z-coordinate"]].to_numpy()
    vals = df[columns].to_numpy()

    interp = scipy.interpolate.LinearNDInterpolator(coords, vals, fill_value=np.nan)

    query_points = np.asarray(query_points)
    n = query_points.shape[0]
    out = np.empty((n, vals.shape[1]), dtype=float)
    for i in range(0, n, batch):
        out[i:i + batch] = interp(query_points[i:i + batch])
    return out


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
# Callable pipeline
# =============================================================================

def run_gpr(
    stl_filepath="input_stls/triangle.stl",
    cfd_filepath="inputs/FLTG.csv",
    stl_scale=1.0 / 1000.0,
    training_point_n_requested=160,
    res=150,
    posterior_batch=4000,
    v_inf=(12.0, 0.0, 0.0),
    n_restarts=6,
    fit_pressure=True,
    verbose=True,
):
    """
    Run the full velocity (and optional pressure) GPR pipeline.

    Returns a dict holding everything the plotting code needs:
        test_points, bounds, res, means_tests, GPR_posterior (Nx3),
        cfd_test_vels, pressure_posterior, training_coords, mesh_vertices,
        fit, metrics.
    """
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

        # --- Panel solver (prior) ---
        solver = FLOWPanelSolver(stl_mesh, v_inf, julia_script="FP.jl",
                                julia_bin="julia", verbose=False)

        def prior_fn(pts):
            return solver.velocity(np.asarray(pts), blank_interior=False).reshape(-1, 3)
        bar()

        bar.text('Sampling...')
        # --- Sample training data ---
        ground_truth, bounds = sample(
        cfd_filepath, stl_mesh, method="drone_array",
        epsilon=0.02, use_signed_distance=True,
        config={"tilt_deg": 30, "n_rows": 10, "n_cols": 10},
        )

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
        means_tests = np.empty((n_test, 3), dtype=float)
        for ci, i in enumerate(range(0, n_test, posterior_batch)):
            chunk = test_points[i:i + posterior_batch]
            means_tests[i:i+posterior_batch] = solver.velocity(chunk, blank_interior=True).reshape(-1, 3)
            if (ci % 50 == 0 or ci == n_chunks - 1):
                print(f"variances chunk {ci + 1}/{n_chunks}", flush=True)

        
        means_training = solver.velocity(training_coords, blank_interior=False).reshape(-1, 3)
        if np.isnan(means_training).any():
            raise RuntimeError("NaNs in direct prior mean at training points.")
        means_training = means_training.reshape(-1, 1)

        prior_train_rmse = rmse(means_training, training_vels)
        bar()

        bar.text('Fit hyperparameters')
        # --- Fit GP to residuals ---
        residuals = training_vels - means_training

        if np.isnan(residuals).any():
            raise RuntimeError("NaNs in residuals_for_fit.")

        fit = fit_hyperparams(training_coords, residuals, n_restarts=n_restarts, jitter=1e-6, seed=0)
        
        print(f"    nll={fit['nll']:.6g}  ell={fit['ell']}  "
            f"var={fit['var']:.4g}  noise={fit['noise']:.4g}", flush=True)

        ell = jnp.asarray(fit["ell"])
        var = float(fit["var"])
        noise = float(fit["noise"])
        bar()

        bar.text('Assemble K + cholesky solve (invert K) at residuals and K(X_train, X_test)...')
        # --- Solve for alpha, build posterior ---
        K = assemble_dat_shi(training_coords, training_coords, ell, var, noise_std=noise, jitter=1e-8)
        c, low = cho_factor(K)
        alpha = cho_solve((c, low), jnp.asarray(residuals)) #inverted K matrix times residuals is alpha
        K_test = assemble_dat_shi(test_points, training_coords, ell, var)
        beta = cho_solve((c, low), jnp.asarray(K_test.T))
        bar()

        bar.text(f"Velocity posterior over grid ({n_chunks} chunks)...")
        GPR_posterior = posterior_mean_batched(
            test_points, training_coords, ell, var, alpha, means_tests,
            batch=posterior_batch, progress_every=50)
        GPR_posterior = np.array(GPR_posterior).reshape(-1, 3)

        bar.text(f"Variance posterior over grid ({n_chunks} chunks)...")
        GPR_variances = posterior_vars_batched(test_points, training_coords, ell, var, beta,
                                               batch=posterior_batch, progress_every=50)
        GPR_variances = np.array(GPR_variances).reshape(-1, 3)

        # Training reconstruction check
        K_signal = assemble_dat_shi(training_coords, training_coords, ell, var, noise_std=0.0, jitter=0.0)
        train_post = jnp.asarray(means_training) + K_signal @ alpha
        posterior_train_rmse = rmse(np.array(train_post), training_vels)
        bar()

        bar.text('Interpolating truth for comparison...')
        # --- CFD truth on test grid (velocity + pressure in one griddata pass) ---
        cfd_test = interpolate_cfd_to_points(
            cfd_filepath, test_points,
            ["x-velocity", "y-velocity", "z-velocity", "pressure"])
        cfd_test_vels = cfd_test[:, :3]
        cfd_p = cfd_test[:, 3]

        valid_cfd = ~np.any(np.isnan(cfd_test_vels), axis=1)
        truth = cfd_test_vels[valid_cfd]
        prior_test_rmse = rmse(means_tests[valid_cfd], truth)
        post_test_rmse = rmse(GPR_posterior[valid_cfd], truth)
        truth_test_rms = float(np.sqrt(np.mean(truth ** 2)))
        bar()

        bar.text('Pressure GPR...')
        # --- Pressure GPR (scalar, sklearn) ---
        pressure_posterior = None
        pressure_test_rmse = None
        if fit_pressure:
            train_p = ground_truth["pressure"].to_numpy()
            p_kernel = Matern(
                length_scale=[1.0, 1.0, 1.0], 
                length_scale_bounds=(1e-2, 1e2), 
                nu=2.5)
                
            p_gpr = GaussianProcessRegressor(
                kernel=p_kernel, normalize_y=True, n_restarts_optimizer=8, random_state=0,
            ).fit(training_coords, train_p)
            print(f"fitted kernel: {p_gpr.kernel_}", flush=True)

            pressure_posterior = predict_batched(p_gpr, test_points, batch=posterior_batch)

            valid_p = ~np.isnan(cfd_p)
            pressure_test_rmse = rmse(pressure_posterior[valid_p], cfd_p[valid_p])
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
        }
        bar.text('Done!')
    

    return {
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
    }


if __name__ == "__main__":
    result = run_gpr(verbose=False)
    print("\nMetrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")
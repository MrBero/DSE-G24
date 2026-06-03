import os

# Force CPU before importing JAX.
os.environ["JAX_PLATFORMS"] = "cpu"

import time
import numpy as np
import scipy.interpolate
import scipy.optimize as spo
import scipy.stats.qmc as qmc
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

plt.style.use("dark_background")
mplstyle.use("fast")

import trimesh

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

jax.config.update("jax_enable_x64", True)

from potential_flow_run import *
from flowpanelwrapper import FLOWPanelSolver
from sampling import sample


# =============================================================================
# Kernel
# =============================================================================

def matern52_np(v1, v2, ell, var):
    """
    Scalar Matérn-5/2 covariance.

    v1, v2: shape (3,)
    ell: shape (3,)
    var: scalar
    """
    diff = v2 - v1

    r = jnp.sqrt(
        (diff[0] / ell[0]) ** 2
        + (diff[1] / ell[1]) ** 2
        + (diff[2] / ell[2]) ** 2
        + 1e-8
    )

    return var * (
        1.0
        + jnp.sqrt(5.0) * r
        + (5.0 / 3.0) * r ** 2
    ) * jnp.exp(-jnp.sqrt(5.0) * r)

@jax.jit
def Hemholtz_K0(V1, V2, ell, var):
    """
    3x3 divergence-free-ish vector kernel block generated from Hessian
    of scalar Matérn-5/2 kernel.
    """
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
    Assemble block vector-valued covariance matrix.

    Ordering:
        point-major vector ordering

        [u_x(p0), u_y(p0), u_z(p0),
         u_x(p1), u_y(p1), u_z(p1),
         ...]

    Returns shape:
        (3 * len(points_1), 3 * len(points_2))
    """
    points_1 = jnp.asarray(points_1)
    points_2 = jnp.asarray(points_2)

    n_1 = points_1.shape[0]
    n_2 = points_2.shape[0]

    blocks = jax.vmap(
        lambda a: jax.vmap(
            lambda b: Hemholtz_K0(a, b, ell, var)
        )(points_2)
    )(points_1)

    result_matrix = jnp.transpose(blocks, (0, 2, 1, 3)).reshape(n_1 * 3, n_2 * 3)

    if n_1 == n_2:
        if noise_std > 0.0:
            result_matrix = result_matrix + (noise_std ** 2 + jitter) * jnp.eye(n_1 * 3)
        elif jitter > 0.0:
            result_matrix = result_matrix + jitter * jnp.eye(n_1 * 3)

    return result_matrix

def fit_hyperparams(
    train_coords,
    train_residuals,
    n_restarts=6,
    jitter=1e-6,
    seed=0,
):
    """
    Fast, reasonably robust bounded GP hyperparameter fit.

    Strategy:
        - Compile the NLL + grad once, reuse across all evaluations.
        - A handful of Latin-hypercube starts + one heuristic start.
        - Short L-BFGS-B polish per start (smooth surface converges fast).
        - Bad points get a finite penalty, never an escalation loop.

    Fits: ell_x, ell_y, ell_z, var, noise
    """
    X = jnp.asarray(train_coords)
    y = jnp.asarray(train_residuals).reshape(-1, 1)
    n = X.shape[0]

    coord_span = np.maximum(np.ptp(train_coords, axis=0), 1e-12)
    volume = float(np.prod(coord_span))
    sample_spacing = float((volume / max(n, 1)) ** (1.0 / 3.0))

    yvar = max(float(np.var(train_residuals)), 1e-12)
    yrms = max(float(np.sqrt(np.mean(train_residuals ** 2))), 1e-12)

    ell_min_scalar = max(sample_spacing / 3.0, 0.25)
    ell_min = np.array([ell_min_scalar] * 3, dtype=float)
    ell_max = 3.0 * coord_span

    var_min, var_max = yvar * 1e-6, yvar * 1e3
    noise_min, noise_max = yrms * 1e-4, yrms * 2.0

    lower = np.log(np.array([*ell_min, var_min, noise_min], dtype=float))
    upper = np.log(np.array([*ell_max, var_max, noise_max], dtype=float))
    bounds = list(zip(lower, upper))

    print("\nHyperparameter bounds")
    print("---------------------")
    print(f"coord_span:      {coord_span}")
    print(f"sample_spacing:  {sample_spacing:.6g}")
    print(f"ell_min:         {ell_min}")
    print(f"ell_max:         {ell_max}")
    print(f"var bounds:      [{var_min:.6g}, {var_max:.6g}]")
    print(f"noise bounds:    [{noise_min:.6g}, {noise_max:.6g}]")

    # NLL compiled ONCE. jitter closed over as a Python float (constant).
    @jax.jit
    def nll(log_theta):
        th = jnp.exp(log_theta)
        ell = th[0:3]
        var = th[3]
        noise = th[4]

        blocks = jax.vmap(
            lambda a: jax.vmap(lambda b: Hemholtz_K0(a, b, ell, var))(X)
        )(X)

        K = jnp.transpose(blocks, (0, 2, 1, 3)).reshape(3 * n, 3 * n)
        K = K + (noise ** 2 + jitter) * jnp.eye(3 * n)

        c, low = cho_factor(K)
        alpha = cho_solve((c, low), y)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(c)))

        return (
            0.5 * (y.T @ alpha)[0, 0]
            + 0.5 * logdet
            + 0.5 * (3 * n) * jnp.log(2.0 * jnp.pi)
        )

    nll_vg = jax.jit(jax.value_and_grad(nll))

    def scipy_fun(log_theta_np):
        val, grad = nll_vg(jnp.asarray(log_theta_np))
        val = float(val)
        grad = np.asarray(grad, dtype=float)
        if not np.isfinite(val) or not np.all(np.isfinite(grad)):
            return 1e12, np.zeros_like(log_theta_np)
        return val, grad

    # Starts: 1 heuristic + (n_restarts-1) Latin-hypercube spread.
    starts = [
        np.log(
            np.array(
                [
                    max(coord_span[0] / 5.0, ell_min[0]),
                    max(coord_span[1] / 5.0, ell_min[1]),
                    max(coord_span[2] / 5.0, ell_min[2]),
                    yvar,
                    0.1 * yrms,
                ],
                dtype=float,
            )
        )
    ]

    n_lhs = max(n_restarts - 1, 0)
    if n_lhs > 0:
        unit = qmc.LatinHypercube(d=5, seed=seed).random(n_lhs)
        starts.extend(list(lower + unit * (upper - lower)))

    # Warm up the JIT once.
    _ = scipy_fun(starts[0])

    best = None
    for k, t0 in enumerate(starts):
        t0 = np.clip(t0, lower, upper)
        res = spo.minimize(
            scipy_fun,
            t0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-8, "gtol": 1e-6},
        )
        theta = np.exp(res.x)
        f = float(res.fun)
        print(
            f"Restart {k}: success={res.success}, nll={f:.6g}, "
            f"ell={theta[0:3]}, var={theta[3]:.6g}, noise={theta[4]:.6g}"
        )
        if np.isfinite(f) and (best is None or f < best[0]):
            best = (f, theta, bool(res.success), str(res.message))

    if best is None:
        raise RuntimeError("Hyperparameter optimization failed for all restarts.")

    f, theta, ok, message = best
    print(f"\nBest nll: {f:.6g} (success={ok})")

    return {
        "ell": theta[0:3],
        "var": float(theta[3]),
        "noise": float(theta[4]),
        "nll": float(f),
        "success": bool(ok),
        "message": message,
        "sample_spacing": sample_spacing,
        "ell_min": ell_min,
        "ell_max": ell_max,
    }

def print_vector_stats(name, arr):
    arr = np.asarray(arr)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"shape:     {arr.shape}")
    print(f"nan count: {np.isnan(arr).sum()}")

    if arr.ndim == 2:
        print(f"min:       {np.nanmin(arr, axis=0)}")
        print(f"max:       {np.nanmax(arr, axis=0)}")
        print(f"mean:      {np.nanmean(arr, axis=0)}")
        print(f"std:       {np.nanstd(arr, axis=0)}")
    else:
        print(f"min:       {np.nanmin(arr)}")
        print(f"max:       {np.nanmax(arr)}")
        print(f"mean:      {np.nanmean(arr)}")
        print(f"std:       {np.nanstd(arr)}")

def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def velocity_magnitude(U):
    return np.sqrt(np.sum(U ** 2, axis=-1))

def interpolate_cfd_velocity_to_points(cfd_filepath, query_points):
    cfd_df = pd.read_csv(cfd_filepath)
    cfd_df.columns = cfd_df.columns.str.strip()

    source_coords = cfd_df[
        ["x-coordinate", "y-coordinate", "z-coordinate"]
    ].to_numpy()

    source_vels = cfd_df[
        ["x-velocity", "y-velocity", "z-velocity"]
    ].to_numpy()

    cfd_query_vels = scipy.interpolate.griddata(
        points=source_coords,
        values=source_vels,
        xi=query_points,
        method="linear",
        fill_value=np.nan,
    )

    return cfd_query_vels

def posterior_mean_batched(
    test_points,
    training_coords,
    ell,
    var,
    alpha,
    means_tests,
    batch=4000,
):
    """
    Stream the GP posterior mean over chunks of test points.

    Avoids materializing the full k_star matrix, which for a 70^3 grid
    and ~150 training points would be huge. Each chunk only ever builds a
    (3*batch, 3*n_train) kernel block.
    """
    test_points = np.asarray(test_points)
    means_tests = np.asarray(means_tests).reshape(-1, 3)

    n_test = test_points.shape[0]
    out = np.empty((n_test, 3), dtype=float)

    alpha_local = jnp.asarray(alpha)
    n_chunks = (n_test + batch - 1) // batch

    for ci, i in enumerate(range(0, n_test, batch)):
        tp = test_points[i:i + batch]

        ks = assemble_dat_shi(
            tp, training_coords, ell, var, noise_std=0.0, jitter=0.0,
        )

        contrib = np.array(ks @ alpha_local).reshape(-1, 3)
        out[i:i + batch] = means_tests[i:i + batch] + contrib

        if (ci % 10 == 0) or (ci == n_chunks - 1):
            print(f"  posterior chunk {ci + 1}/{n_chunks}", flush=True)

    return out.reshape(-1, 1)

def main():
    # -------------------------------------------------------------------------
    # User settings
    # -------------------------------------------------------------------------
    stl_filepath = "input_stls/triangle.stl"
    cfd_filepath = "inputs/FLTG.csv"

    # If STL is in mm and CFD CSV is in m, use 1/1000.
    # If STL is already in m, use 1.0.
    STL_SCALE = 1.0 / 1000.0

    training_point_n_requested = 150
    res = 100

    # Memory control: chunk size for streaming the posterior over test points.
    posterior_batch = 4000

    V_inf = np.array([12.0, 0.0, 0.0])

    # Plot slice through object, not the middle of the full CFD box.
    z_slice_target = 2.5

    # -------------------------------------------------------------------------
    # Load mesh and scale once
    # -------------------------------------------------------------------------
    stl_mesh = trimesh.load_mesh(stl_filepath)

    raw_bounds = stl_mesh.bounds.copy()
    raw_extents = stl_mesh.extents.copy()

    if STL_SCALE != 1.0:
        stl_mesh.apply_scale(STL_SCALE)

    print("\nRaw STL mesh")
    print("------------")
    print(f"bounds:\n{raw_bounds}")
    print(f"extents: {raw_extents}")

    print("\nScaled STL mesh")
    print("---------------")
    print(f"STL_SCALE: {STL_SCALE}")
    print(f"bounds:\n{stl_mesh.bounds}")
    print(f"extents: {stl_mesh.extents}")

    # -------------------------------------------------------------------------
    # Build the panel solver FIRST so it can act as the sampling prior.
    # -------------------------------------------------------------------------
    surface_source_solver = FLOWPanelSolver(
        stl_mesh,
        V_inf,
        julia_script="FP.jl",
        julia_bin="julia",
        verbose=True,
    )

    def prior_fn(pts):
        """Potential-flow prior velocity at arbitrary points (M,3)->(M,3)."""
        return surface_source_solver.velocity(
            np.asarray(pts), blank_interior=False
        ).reshape(-1, 3)

    # -------------------------------------------------------------------------
    # Sample training data (residual-adaptive CV sampling around the body)
    # -------------------------------------------------------------------------
    ground_truth, bounds = sample(
        cfd_filepath,
        stl_mesh,
        method="cv",
        num_samples=training_point_n_requested,
        epsilon=0.02,
        use_signed_distance=True,
        max_points=150,
        prior_fn=prior_fn,
    )

    training_coords = ground_truth[
        ["x-target", "y-target", "z-target"]
    ].to_numpy()

    training_vels_matrix = ground_truth[
        ["x-velocity", "y-velocity", "z-velocity"]
    ].to_numpy()

    training_vels = training_vels_matrix.reshape(-1, 1)
    training_point_n = len(ground_truth)

    print("\nCFD bounds")
    print("----------")
    print(bounds)
    print(f"CFD extents: {bounds[:, 1] - bounds[:, 0]}")

    print("\nMesh-vs-CFD scale check")
    print("-----------------------")
    print(f"Mesh extents: {stl_mesh.extents}")
    print(f"CFD extents:  {bounds[:, 1] - bounds[:, 0]}")
    with np.errstate(divide="ignore", invalid="ignore"):
        print(f"CFD / mesh extent ratio: {(bounds[:, 1] - bounds[:, 0]) / stl_mesh.extents}")

    print(f"\nRequested training points: {training_point_n_requested}")
    print(f"Actual training points:    {training_point_n}")

    try:
        inside_training = stl_mesh.contains(training_coords)
        print(f"Training points inside mesh: {inside_training.sum()} / {training_point_n}")
    except Exception as exc:
        print(f"Could not check training points inside mesh: {exc}")

    print_vector_stats("training_coords", training_coords)
    print_vector_stats("training_vels_matrix", training_vels_matrix)

    if training_point_n == 0:
        raise RuntimeError("No training points were sampled.")

    # -------------------------------------------------------------------------
    # Build test grid
    # -------------------------------------------------------------------------
    x, y, z = np.meshgrid(
        np.linspace(bounds[0, 0], bounds[0, 1], res),
        np.linspace(bounds[1, 0], bounds[1, 1], res),
        np.linspace(bounds[2, 0], bounds[2, 1], res),
        indexing="ij",
    )

    test_points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    test_point_n = test_points.shape[0]

    print(f"\nTotal number of test points: {test_point_n}")
    print_vector_stats("test_points", test_points)

    # -------------------------------------------------------------------------
    # Prior mean from panel solver
    # -------------------------------------------------------------------------
    tick = time.perf_counter()

    potential_flow_field = surface_source_solver.generate_flow_field(
        x, y, z, zero_inside=True,
    )

    means_tests = potential_flow_field.reshape(-1, 3)

    # Direct evaluation at training points. Do not interpolate from test grid.
    means_training_matrix = surface_source_solver.velocity(
        training_coords, blank_interior=False,
    ).reshape(-1, 3)

    if np.isnan(means_training_matrix).any():
        raise RuntimeError(
            f"NaNs in direct prior mean at training points: "
            f"{np.isnan(means_training_matrix).sum()}"
        )

    means_training = means_training_matrix.reshape(-1, 1)

    tock = time.perf_counter()
    print(f"\nPrior means (julia script) calculated in: {tock - tick:.3f}s")

    print_vector_stats("means_training", means_training)

    prior_train_rmse = rmse(means_training, training_vels)
    data_rms = float(np.sqrt(np.mean(training_vels ** 2)))

    print("\nPrior quality at training points")
    print("--------------------------------")
    print(f"Prior RMSE:          {prior_train_rmse:.6g}")
    print(f"Training data RMS:   {data_rms:.6g}")
    print(f"Relative prior RMSE: {prior_train_rmse / max(data_rms, 1e-12):.6g}")

    # -------------------------------------------------------------------------
    # Fit GP to residuals
    # -------------------------------------------------------------------------
    residuals_for_fit = training_vels - means_training

    if np.isnan(residuals_for_fit).any():
        raise RuntimeError(
            f"NaNs in residuals_for_fit: {np.isnan(residuals_for_fit).sum()}"
        )

    print_vector_stats("residuals_for_fit", residuals_for_fit)

    tick = time.perf_counter()
    fit = fit_hyperparams(
        training_coords,
        residuals_for_fit,
        n_restarts=6,
        jitter=1e-6,
        seed=0,
    )
    tock = time.perf_counter()

    print(f"\nHyperparameter fitting complete in {tock - tick:.3f}s")
    # print(f"fit: {fit}")

    ell = jnp.asarray(fit["ell"])
    var = float(fit["var"])
    noise = float(fit["noise"])

    dx_test = (bounds[:, 1] - bounds[:, 0]) / (res - 1)

    print("\nLengthscales")
    print("------------------------")
    print(f"Test grid spacing: {dx_test}")
    print(f"Fitted ell:        {fit['ell']}")
    print(f"ell / dx_test:     {fit['ell'] / dx_test}")
    print(f"Sample spacing:    {fit.get('sample_spacing', np.nan)}")

    # -------------------------------------------------------------------------
    # Assemble training covariance and solve for alpha
    # -------------------------------------------------------------------------
    tick = time.perf_counter()
    K_matrix = assemble_dat_shi(training_coords, training_coords, ell, var, noise_std=noise, jitter=1e-8)
    tock = time.perf_counter()

    print(f"\nK_matrix assembled in {tock - tick:.3f}s")
    print(f"K_matrix shape: {K_matrix.shape}")

    residuals = jnp.asarray(training_vels - means_training).reshape(-1, 1)
    c, low = cho_factor(K_matrix)
    alpha = cho_solve((c, low), residuals)

    tick = time.perf_counter()
    GPR_posterior = posterior_mean_batched(test_points, training_coords, ell, var, alpha, means_tests, batch=posterior_batch)
    tock = time.perf_counter()

    print(f"\nGPR posterior generated in {tock - tick:.3f}s")
    print(f"GPR_posterior shape: {GPR_posterior.shape}")

    GPR_posterior_reshaped = np.array(GPR_posterior).reshape(-1, 3)

    print("\nTraining reconstruction check")
    print("-----------------------------")

    K_train_signal = assemble_dat_shi(training_coords, training_coords, ell, var, noise_std=0.0, jitter=0.0)

    train_posterior = jnp.asarray(means_training) + K_train_signal @ alpha

    posterior_train_rmse = rmse(np.array(train_posterior), training_vels)

    print(f"Prior train RMSE:     {prior_train_rmse:.6g}")
    print(f"Posterior train RMSE: {posterior_train_rmse:.6g}")
    print(f"Fitted ell:           {fit['ell']}")
    print(f"Fitted var:           {fit['var']:.6g}")
    print(f"Fitted noise:         {fit['noise']:.6g}")

    print("\nInterpolating CFD truth onto test grid...")
    cfd_test_vels = interpolate_cfd_velocity_to_points(cfd_filepath, test_points)

    valid_cfd = ~np.any(np.isnan(cfd_test_vels), axis=1)

    prior_test = means_tests[valid_cfd]
    post_test = GPR_posterior_reshaped[valid_cfd]
    truth_test = cfd_test_vels[valid_cfd]

    prior_test_rmse = rmse(prior_test, truth_test)
    post_test_rmse = rmse(post_test, truth_test)
    truth_test_rms = float(np.sqrt(np.mean(truth_test ** 2)))

    print("\nTest-grid CFD comparison")
    print("------------------------")
    print(f"Valid CFD test points:        {valid_cfd.sum()} / {len(valid_cfd)}")
    print(f"Prior test RMSE:              {prior_test_rmse:.6g}")
    print(f"Posterior test RMSE:          {post_test_rmse:.6g}")
    print(f"Truth test RMS:               {truth_test_rms:.6g}")
    print(f"Relative prior test RMSE:     {prior_test_rmse / max(truth_test_rms, 1e-12):.6g}")
    print(f"Relative posterior test RMSE: {post_test_rmse / max(truth_test_rms, 1e-12):.6g}")
    
    GPR_vel_mags = np.sqrt(np.sum(GPR_posterior_reshaped ** 2, axis=1))

    print_vector_stats("GPR_posterior_reshaped", GPR_posterior_reshaped)
    print_vector_stats("GPR_vel_mags", GPR_vel_mags)

    fig1 = plt.figure(figsize=(9, 7))
    ax1 = fig1.add_subplot(projection="3d")

    sc = ax1.scatter3D(test_points[:, 0], test_points[:, 1], test_points[:, 2], c=GPR_vel_mags, alpha=0.2, s=4)

    fig1.colorbar(sc, ax=ax1, label="|velocity|")

    verts = surface_source_solver.mesh.vertices

    ax1.scatter3D(verts[:, 0], verts[:, 1], verts[:, 2], c="black", s=4)

    # Overlay actual training sample locations.
    ax1.scatter3D(training_coords[:, 0], training_coords[:, 1], training_coords[:, 2], c="red", s=18, depthshade=False, label="training samples")
    ax1.legend()

    ax1.set_title("GPR posterior velocity magnitude")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_aspect("equal")

    P = test_points.reshape(res, res, res, 3)

    prior_U = means_tests.reshape(res, res, res, 3)
    post_U = GPR_posterior_reshaped.reshape(res, res, res, 3)
    cfd_U = cfd_test_vels.reshape(res, res, res, 3)

    z_grid = np.linspace(bounds[2, 0], bounds[2, 1], res)
    k = int(np.argmin(np.abs(z_grid - z_slice_target)))
    z_here = z_grid[k]

    print(f"\nPlotting xy slices at z={z_here:.6g}")

    Xs = P[:, :, k, 0]
    Ys = P[:, :, k, 1]

    prior_slice = prior_U[:, :, k, :]
    post_slice = post_U[:, :, k, :]
    cfd_slice = cfd_U[:, :, k, :]

    prior_mag = velocity_magnitude(prior_slice)
    post_mag = velocity_magnitude(post_slice)
    cfd_mag = velocity_magnitude(cfd_slice)

    prior_us = prior_slice[:, :, 0]
    prior_vs = prior_slice[:, :, 1]

    post_us = post_slice[:, :, 0]
    post_vs = post_slice[:, :, 1]

    cfd_us = cfd_slice[:, :, 0]
    cfd_vs = cfd_slice[:, :, 1]

    # Mesh vertices near the actual z slice.
    slice_tol = 0.5 * abs(z_grid[1] - z_grid[0]) if res > 1 else 1e-6
    near_slice = np.abs(verts[:, 2] - z_here) <= slice_tol

    print(f"Mesh vertices near slice: {near_slice.sum()} / {len(verts)}")

    # -------------------------------------------------------------------------
    # CFD vs prior vs posterior
    # -------------------------------------------------------------------------
    prior_err = prior_mag - cfd_mag
    post_err = post_mag - cfd_mag

    fig_cfd, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    mag_vmin = np.nanmin([cfd_mag, prior_mag, post_mag])
    mag_vmax = np.nanmax([cfd_mag, prior_mag, post_mag])

    err_lim = np.nanmax(np.abs([prior_err, post_err]))

    plot_items = [
        (axs[0, 0], cfd_mag, "CFD truth |u|", "viridis", mag_vmin, mag_vmax),
        (axs[0, 1], prior_mag, "Prior |u|", "viridis", mag_vmin, mag_vmax),
        (axs[0, 2], post_mag, "Posterior |u|", "viridis", mag_vmin, mag_vmax),
        (axs[1, 0], post_mag - prior_mag, "Posterior - Prior |u|", "coolwarm", None, None),
        (axs[1, 1], prior_err, "Prior - CFD |u|", "coolwarm", -err_lim, err_lim),
        (axs[1, 2], post_err, "Posterior - CFD |u|", "coolwarm", -err_lim, err_lim),
    ]

    pp_lim = np.nanmax(np.abs(post_mag - prior_mag))

    for ax, field, title, cmap, lo, hi in plot_items:
        if title.startswith("Posterior - Prior"):
            lo = -pp_lim
            hi = pp_lim

        pc = ax.contourf(Xs, Ys, field, levels=30, cmap=cmap, vmin=lo, vmax=hi)
        fig_cfd.colorbar(pc, ax=ax)

        if near_slice.any():
            ax.scatter(verts[near_slice, 0], verts[near_slice, 1], c="black", s=8)

        ax.set_aspect("equal")
        ax.set_title(f"{title}, z={z_here:.4g}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    plt.show()

if __name__ == "__main__":
    main() 
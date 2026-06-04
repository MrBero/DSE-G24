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

def plot_sampling_layout(
    stl_mesh,
    samples_df,
    bounds,
    wind_direction,
    savepath="sampling_layout.png",
    title="Training sample layout",
):
    """Plot where the selected training measurements are, colored by group."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    coords = samples_df[
        ["x-target", "y-target", "z-target"]
    ].to_numpy(float)

    if "sample-group" in samples_df.columns:
        groups = samples_df["sample-group"].astype(str).to_numpy()
    else:
        groups = np.full(len(coords), "training samples", dtype=object)

    group_order = [
        "wake core",
        "wake shear",
        "wake recovery",
        "near-body guards",
        "freestream anchors",
        "cv windward face",
        "cv leeward face",
        "cv side/top faces",
        "separation guards",
        "upstream anchors",
        "separation shell",
        "wake",
        "windward/edge",
        "upstream",
        "wake fill",
        "drone-safe fill",
        "domain fill",
        "training samples",
    ]
    colors = {
        "wake core": "#2f80ed",
        "wake shear": "#00d4ff",
        "wake recovery": "#8e5cf7",
        "near-body guards": "#ff9f1c",
        "freestream anchors": "#06d6a0",
        "cv windward face": "#ff5c5c",
        "cv leeward face": "#a855f7",
        "cv side/top faces": "#ffd166",
        "separation guards": "#f72585",
        "upstream anchors": "#06d6a0",
        "separation shell": "#ff5c5c",
        "wake": "#2f80ed",
        "windward/edge": "#ffd166",
        "upstream": "#06d6a0",
        "wake fill": "#7aa6ff",
        "drone-safe fill": "#b388ff",
        "domain fill": "#8d99ae",
        "training samples": "#ff5c5c",
    }

    present = [g for g in group_order if np.any(groups == g)]
    present += sorted(set(groups) - set(present))

    fig = plt.figure(figsize=(15, 6), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axxy = fig.add_subplot(1, 2, 2)

    tris = np.asarray(stl_mesh.triangles)
    mesh_collection = Poly3DCollection(
        tris,
        facecolor=(0.55, 0.55, 0.55, 0.22),
        edgecolor=(0.05, 0.05, 0.05, 0.35),
        linewidth=0.25,
    )
    ax3d.add_collection3d(mesh_collection)

    for group in present:
        mask = groups == group
        count = int(mask.sum())
        pct = 100.0 * count / max(len(coords), 1)
        label = f"{group}: {count} ({pct:.1f}%)"
        color = colors.get(group, "#ffffff")
        ax3d.scatter(
            coords[mask, 0],
            coords[mask, 1],
            coords[mask, 2],
            s=32,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            depthshade=False,
            label=label,
        )
        axxy.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=32,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            label=label,
        )

    verts = np.asarray(stl_mesh.vertices)
    axxy.scatter(verts[:, 0], verts[:, 1], c="0.6", s=6, alpha=0.8)

    center = coords.mean(axis=0)
    flow = np.asarray(wind_direction, dtype=float)
    flow_norm = np.linalg.norm(flow)
    if flow_norm > 0:
        flow = flow / flow_norm
        arrow_len = 0.15 * (bounds[0, 1] - bounds[0, 0])
        axxy.arrow(
            center[0],
            bounds[1, 0] + 0.12 * (bounds[1, 1] - bounds[1, 0]),
            arrow_len * flow[0],
            arrow_len * flow[1],
            color="white",
            width=0.035,
            head_width=0.35,
            length_includes_head=True,
        )

    ax3d.set_title(title)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_xlim(bounds[0])
    ax3d.set_ylim(bounds[1])
    ax3d.set_zlim(bounds[2])
    ax3d.set_box_aspect(bounds[:, 1] - bounds[:, 0])

    axxy.set_title(f"{title}, top view")
    axxy.set_xlabel("x")
    axxy.set_ylabel("y")
    axxy.set_xlim(bounds[0])
    axxy.set_ylim(bounds[1])
    axxy.set_aspect("equal", adjustable="box")
    axxy.grid(alpha=0.18)

    ax3d.legend(loc="upper left", fontsize=8)
    axxy.legend(loc="upper right", fontsize=8)
    fig.savefig(savepath, dpi=160, bbox_inches="tight")
    print(f"\nSampling layout figure saved to: {savepath}")
    return fig

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


def plot_flow_reconstruction_slice(
    method_name,
    stl_mesh,
    bounds,
    res,
    test_points,
    prior_values,
    posterior_values,
    cfd_values,
    training_coords,
    z_slice_target,
    savepath,
):
    """Save a CFD/prior/posterior slice comparison for one sampling method."""
    verts = np.asarray(stl_mesh.vertices)
    P = test_points.reshape(res, res, res, 3)

    prior_U = prior_values.reshape(res, res, res, 3)
    post_U = posterior_values.reshape(res, res, res, 3)
    cfd_U = cfd_values.reshape(res, res, res, 3)

    z_grid = np.linspace(bounds[2, 0], bounds[2, 1], res)
    k = int(np.argmin(np.abs(z_grid - z_slice_target)))
    z_here = z_grid[k]

    print(f"\nPlotting {method_name} xy slices at z={z_here:.6g}")

    Xs = P[:, :, k, 0]
    Ys = P[:, :, k, 1]

    prior_mag = velocity_magnitude(prior_U[:, :, k, :])
    post_mag = velocity_magnitude(post_U[:, :, k, :])
    cfd_mag = velocity_magnitude(cfd_U[:, :, k, :])

    slice_tol = 0.5 * abs(z_grid[1] - z_grid[0]) if res > 1 else 1e-6
    near_slice = np.abs(verts[:, 2] - z_here) <= slice_tol
    sample_near_slice = np.abs(training_coords[:, 2] - z_here) <= slice_tol

    print(f"Mesh vertices near slice: {near_slice.sum()} / {len(verts)}")
    print(
        f"{method_name} training samples near slice: "
        f"{sample_near_slice.sum()} / {len(training_coords)}"
    )

    prior_err = prior_mag - cfd_mag
    post_err = post_mag - cfd_mag

    fig_cfd, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    mag_vmin = np.nanmin([cfd_mag, prior_mag, post_mag])
    mag_vmax = np.nanmax([cfd_mag, prior_mag, post_mag])
    err_lim = np.nanmax(np.abs([prior_err, post_err]))
    pp_lim = np.nanmax(np.abs(post_mag - prior_mag))

    plot_items = [
        (axs[0, 0], cfd_mag, "CFD truth |u|", "viridis", mag_vmin, mag_vmax),
        (axs[0, 1], prior_mag, "Prior |u|", "viridis", mag_vmin, mag_vmax),
        (axs[0, 2], post_mag, "Posterior |u|", "viridis", mag_vmin, mag_vmax),
        (
            axs[1, 0],
            post_mag - prior_mag,
            "Posterior - Prior |u|",
            "coolwarm",
            -pp_lim,
            pp_lim,
        ),
        (axs[1, 1], prior_err, "Prior - CFD |u|", "coolwarm", -err_lim, err_lim),
        (axs[1, 2], post_err, "Posterior - CFD |u|", "coolwarm", -err_lim, err_lim),
    ]

    for ax, field, title, cmap, lo, hi in plot_items:
        pc = ax.contourf(Xs, Ys, field, levels=30, cmap=cmap, vmin=lo, vmax=hi)
        fig_cfd.colorbar(pc, ax=ax)

        if near_slice.any():
            ax.scatter(verts[near_slice, 0], verts[near_slice, 1], c="black", s=8)
        if sample_near_slice.any():
            ax.scatter(
                training_coords[sample_near_slice, 0],
                training_coords[sample_near_slice, 1],
                c="white",
                s=18,
                marker="x",
                linewidth=0.8,
            )

        ax.set_aspect("equal")
        ax.set_title(f"{method_name}: {title}, z={z_here:.4g}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig_cfd.savefig(savepath, dpi=160, bbox_inches="tight")
    print(f"Flow reconstruction figure saved to: {savepath}")
    return fig_cfd


def main():
    # -------------------------------------------------------------------------
    # User settings
    # -------------------------------------------------------------------------
    stl_filepath = "input_stls/triangle.stl"
    cfd_filepath = "inputs/FLTG.csv"

    # If STL is in mm and CFD CSV is in m, use 1/1000.
    # If STL is already in m, use 1.0.
    STL_SCALE = 1.0 / 1000.0

    training_point_n_requested = 120
    RECONSTRUCTION_METHODS = ("wake_rmse", "force_cv")
    res = 100

    # Memory control: chunk size for streaming the posterior over test points.
    posterior_batch = 4000

    V_inf = np.array([12.0, 0.0, 0.0])
    julia_bin = os.path.abspath(
        os.path.join(".tools", "julia-1.10.11", "bin", "julia.exe")
    )

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

    try:
        stl_mesh.contains(np.asarray(stl_mesh.bounds).mean(axis=0).reshape(1, 3))
    except Exception as exc:
        raise RuntimeError(
            "Mesh interior masking requires trimesh's contains() query. "
            "Install the missing spatial-index dependency with: pip install rtree"
        ) from exc

    # -------------------------------------------------------------------------
    # Build the panel solver FIRST so it can act as the sampling prior.
    # -------------------------------------------------------------------------
    surface_source_solver = FLOWPanelSolver(
        stl_mesh,
        V_inf,
        julia_script="FP.jl",
        julia_bin=julia_bin,
        verbose=True,
    )

    def prior_fn(pts):
        """Potential-flow prior velocity at arbitrary points (M,3)->(M,3)."""
        return surface_source_solver.velocity(
            np.asarray(pts), blank_interior=False
        ).reshape(-1, 3)

    # -------------------------------------------------------------------------
    # Sample and plot each reconstruction objective.
    # -------------------------------------------------------------------------
    training_sets = {}
    bounds = None

    for method_name in RECONSTRUCTION_METHODS:
        print(f"\nPreparing sampling strategy: {method_name}")
        print("=" * (29 + len(method_name)))

        ground_truth, method_bounds = sample(
            cfd_filepath,
            stl_mesh,
            method=method_name,
            num_samples=training_point_n_requested,
            epsilon=0.02,
            use_signed_distance=True,
            max_points=120,
            wind_direction=V_inf,
            shell_offsets=(0.50, 0.75, 1.00),
            min_drone_clearance=0.50,
            min_measurement_spacing=0.50,
            prior_fn=prior_fn,
        )

        if bounds is None:
            bounds = method_bounds
        elif not np.allclose(bounds, method_bounds):
            raise RuntimeError(
                f"CFD bounds changed between sampling methods: "
                f"{bounds} vs {method_bounds}"
            )

        training_coords = ground_truth[
            ["x-target", "y-target", "z-target"]
        ].to_numpy()
        training_vels_matrix = ground_truth[
            ["x-velocity", "y-velocity", "z-velocity"]
        ].to_numpy()
        training_vels = training_vels_matrix.reshape(-1, 1)
        training_point_n = len(ground_truth)

        print(f"\n{method_name} requested training points: {training_point_n_requested}")
        print(f"{method_name} actual training points:    {training_point_n}")

        try:
            inside_training = stl_mesh.contains(training_coords)
            print(
                f"{method_name} training points inside mesh: "
                f"{inside_training.sum()} / {training_point_n}"
            )
        except Exception as exc:
            print(f"Could not check {method_name} training points inside mesh: {exc}")

        print_vector_stats(f"{method_name} training_coords", training_coords)
        print_vector_stats(f"{method_name} training_vels_matrix", training_vels_matrix)

        if training_point_n == 0:
            raise RuntimeError(f"No training points were sampled for {method_name}.")

        fig = plot_sampling_layout(
            stl_mesh,
            ground_truth,
            bounds,
            V_inf,
            savepath=f"sampling_layout_{method_name}.png",
            title=f"{method_name} sampling",
        )
        if method_name == RECONSTRUCTION_METHODS[-1]:
            fig.savefig("sampling_layout.png", dpi=160, bbox_inches="tight")
            print("Sampling layout figure saved to: sampling_layout.png")

        training_sets[method_name] = {
            "ground_truth": ground_truth,
            "training_coords": training_coords,
            "training_vels_matrix": training_vels_matrix,
            "training_vels": training_vels,
        }

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

    means_tests_base = potential_flow_field.reshape(-1, 3)
    inside_test = stl_mesh.contains(test_points)
    means_tests_base[inside_test] = 0.0

    tock = time.perf_counter()
    print(f"\nPrior means (julia script) calculated in: {tock - tick:.3f}s")
    print(
        f"Zeroed prior velocity inside mesh: "
        f"{inside_test.sum()} / {len(inside_test)} test points"
    )

    print("\nInterpolating CFD truth onto test grid...")
    cfd_test_vels = interpolate_cfd_velocity_to_points(cfd_filepath, test_points)
    cfd_test_vels[inside_test] = np.nan

    valid_cfd = (~np.any(np.isnan(cfd_test_vels), axis=1)) & (~inside_test)
    truth_test = cfd_test_vels[valid_cfd]
    truth_test_rms = float(np.sqrt(np.mean(truth_test ** 2)))
    dx_test = (bounds[:, 1] - bounds[:, 0]) / (res - 1)

    print("\nCommon test-grid CFD data")
    print("-------------------------")
    print(f"Valid CFD test points: {valid_cfd.sum()} / {len(valid_cfd)}")
    print(f"Truth test RMS:        {truth_test_rms:.6g}")
    print(f"Test grid spacing:     {dx_test}")

    reconstruction_summaries = []

    for method_name, data in training_sets.items():
        print(f"\nRunning GPR reconstruction: {method_name}")
        print("=" * (28 + len(method_name)))

        training_coords = data["training_coords"]
        training_vels = data["training_vels"]

        means_training_matrix = surface_source_solver.velocity(
            training_coords, blank_interior=False,
        ).reshape(-1, 3)

        if np.isnan(means_training_matrix).any():
            raise RuntimeError(
                f"NaNs in direct prior mean at {method_name} training points: "
                f"{np.isnan(means_training_matrix).sum()}"
            )

        means_training = means_training_matrix.reshape(-1, 1)
        print_vector_stats(f"{method_name} means_training", means_training)

        prior_train_rmse = rmse(means_training, training_vels)
        data_rms = float(np.sqrt(np.mean(training_vels ** 2)))

        print(f"\n{method_name} prior quality at training points")
        print("-" * (len(method_name) + 33))
        print(f"Prior RMSE:          {prior_train_rmse:.6g}")
        print(f"Training data RMS:   {data_rms:.6g}")
        print(f"Relative prior RMSE: {prior_train_rmse / max(data_rms, 1e-12):.6g}")

        residuals_for_fit = training_vels - means_training
        if np.isnan(residuals_for_fit).any():
            raise RuntimeError(
                f"NaNs in {method_name} residuals_for_fit: "
                f"{np.isnan(residuals_for_fit).sum()}"
            )
        print_vector_stats(f"{method_name} residuals_for_fit", residuals_for_fit)

        tick = time.perf_counter()
        fit = fit_hyperparams(
            training_coords,
            residuals_for_fit,
            n_restarts=6,
            jitter=1e-6,
            seed=0,
        )
        tock = time.perf_counter()

        print(f"\n{method_name} hyperparameter fitting complete in {tock - tick:.3f}s")

        ell = jnp.asarray(fit["ell"])
        var = float(fit["var"])
        noise = float(fit["noise"])

        print(f"\n{method_name} lengthscales")
        print("-" * (len(method_name) + 13))
        print(f"Test grid spacing: {dx_test}")
        print(f"Fitted ell:        {fit['ell']}")
        print(f"ell / dx_test:     {fit['ell'] / dx_test}")
        print(f"Sample spacing:    {fit.get('sample_spacing', np.nan)}")

        tick = time.perf_counter()
        K_matrix = assemble_dat_shi(
            training_coords,
            training_coords,
            ell,
            var,
            noise_std=noise,
            jitter=1e-8,
        )
        tock = time.perf_counter()

        print(f"\n{method_name} K_matrix assembled in {tock - tick:.3f}s")
        print(f"K_matrix shape: {K_matrix.shape}")

        residuals = jnp.asarray(training_vels - means_training).reshape(-1, 1)
        c, low = cho_factor(K_matrix)
        alpha = cho_solve((c, low), residuals)

        tick = time.perf_counter()
        GPR_posterior = posterior_mean_batched(
            test_points,
            training_coords,
            ell,
            var,
            alpha,
            means_tests_base,
            batch=posterior_batch,
        )
        tock = time.perf_counter()

        print(f"\n{method_name} GPR posterior generated in {tock - tick:.3f}s")
        print(f"GPR_posterior shape: {GPR_posterior.shape}")

        GPR_posterior_reshaped = np.array(GPR_posterior).reshape(-1, 3)
        GPR_posterior_reshaped[inside_test] = 0.0
        print(
            f"Zeroed {method_name} posterior velocity inside mesh: "
            f"{inside_test.sum()} / {len(inside_test)} test points"
        )

        print(f"\n{method_name} training reconstruction check")
        print("-" * (len(method_name) + 30))

        K_train_signal = assemble_dat_shi(
            training_coords,
            training_coords,
            ell,
            var,
            noise_std=0.0,
            jitter=0.0,
        )
        train_posterior = jnp.asarray(means_training) + K_train_signal @ alpha
        posterior_train_rmse = rmse(np.array(train_posterior), training_vels)

        print(f"Prior train RMSE:     {prior_train_rmse:.6g}")
        print(f"Posterior train RMSE: {posterior_train_rmse:.6g}")
        print(f"Fitted ell:           {fit['ell']}")
        print(f"Fitted var:           {fit['var']:.6g}")
        print(f"Fitted noise:         {fit['noise']:.6g}")

        prior_test = means_tests_base[valid_cfd]
        post_test = GPR_posterior_reshaped[valid_cfd]

        prior_test_rmse = rmse(prior_test, truth_test)
        post_test_rmse = rmse(post_test, truth_test)

        print(f"\n{method_name} test-grid CFD comparison")
        print("-" * (len(method_name) + 25))
        print(f"Valid CFD test points:        {valid_cfd.sum()} / {len(valid_cfd)}")
        print(f"Prior test RMSE:              {prior_test_rmse:.6g}")
        print(f"Posterior test RMSE:          {post_test_rmse:.6g}")
        print(f"Truth test RMS:               {truth_test_rms:.6g}")
        print(
            f"Relative prior test RMSE:     "
            f"{prior_test_rmse / max(truth_test_rms, 1e-12):.6g}"
        )
        print(
            f"Relative posterior test RMSE: "
            f"{post_test_rmse / max(truth_test_rms, 1e-12):.6g}"
        )

        GPR_vel_mags = np.sqrt(np.sum(GPR_posterior_reshaped ** 2, axis=1))
        print_vector_stats(
            f"{method_name} GPR_posterior_reshaped",
            GPR_posterior_reshaped,
        )
        print_vector_stats(f"{method_name} GPR_vel_mags", GPR_vel_mags)

        plot_flow_reconstruction_slice(
            method_name,
            stl_mesh,
            bounds,
            res,
            test_points,
            means_tests_base,
            GPR_posterior_reshaped,
            cfd_test_vels,
            training_coords,
            z_slice_target,
            savepath=f"flow_reconstruction_{method_name}.png",
        )

        reconstruction_summaries.append(
            {
                "method": method_name,
                "posterior_train_rmse": posterior_train_rmse,
                "posterior_test_rmse": post_test_rmse,
                "relative_posterior_test_rmse": (
                    post_test_rmse / max(truth_test_rms, 1e-12)
                ),
            }
        )

    print("\nReconstruction summary")
    print("----------------------")
    for item in reconstruction_summaries:
        print(
            f"{item['method']}: "
            f"train RMSE {item['posterior_train_rmse']:.6g}, "
            f"test RMSE {item['posterior_test_rmse']:.6g}, "
            f"relative test RMSE {item['relative_posterior_test_rmse']:.6g}"
        )
    plt.show()

if __name__ == "__main__":
    main() 

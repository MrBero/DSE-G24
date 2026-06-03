import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

plt.style.use("dark_background")
mplstyle.use("fast")


def _velocity_magnitude(U):
    return np.sqrt(np.sum(U ** 2, axis=-1))


def plot_posterior_3d(result):
    """3D scatter of posterior velocity magnitude with mesh + training points."""
    test_points = result["test_points"]
    GPR_posterior = result["GPR_posterior"]
    verts = result["mesh_vertices"]
    training_coords = result["training_coords"]

    vel_mags = _velocity_magnitude(GPR_posterior)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")

    sc = ax.scatter3D(test_points[:, 0], test_points[:, 1], test_points[:, 2],
                      c=vel_mags, alpha=0.2, s=4)
    fig.colorbar(sc, ax=ax, label="|velocity|")

    ax.scatter3D(verts[:, 0], verts[:, 1], verts[:, 2], c="black", s=4)
    ax.scatter3D(training_coords[:, 0], training_coords[:, 1], training_coords[:, 2],
                 c="red", s=18, depthshade=False, label="training samples")
    ax.legend()

    ax.set_title("GPR posterior velocity magnitude")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_aspect("equal")
    return fig


def plot_slice_comparison(result, z_slice_target=2.5):
    """2x3 panel of CFD / prior / posterior |u| and their differences at a z-slice."""
    res = result["res"]
    bounds = result["bounds"]
    test_points = result["test_points"]
    verts = result["mesh_vertices"]

    P = test_points.reshape(res, res, res, 3)
    prior_U = result["means_tests"].reshape(res, res, res, 3)
    post_U = result["GPR_posterior"].reshape(res, res, res, 3)
    cfd_U = result["cfd_test_vels"].reshape(res, res, res, 3)

    z_grid = np.linspace(bounds[2, 0], bounds[2, 1], res)
    k = int(np.argmin(np.abs(z_grid - z_slice_target)))
    z_here = z_grid[k]

    Xs = P[:, :, k, 0]
    Ys = P[:, :, k, 1]

    prior_mag = _velocity_magnitude(prior_U[:, :, k, :])
    post_mag = _velocity_magnitude(post_U[:, :, k, :])
    cfd_mag = _velocity_magnitude(cfd_U[:, :, k, :])

    slice_tol = 0.5 * abs(z_grid[1] - z_grid[0]) if res > 1 else 1e-6
    near_slice = np.abs(verts[:, 2] - z_here) <= slice_tol

    prior_err = prior_mag - cfd_mag
    post_err = post_mag - cfd_mag

    fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    mag_vmin = np.nanmin([cfd_mag, prior_mag, post_mag])
    mag_vmax = np.nanmax([cfd_mag, prior_mag, post_mag])
    err_lim = np.nanmax(np.abs([prior_err, post_err]))
    pp_lim = np.nanmax(np.abs(post_mag - prior_mag))

    plot_items = [
        (axs[0, 0], cfd_mag, "CFD truth |u|", "viridis", mag_vmin, mag_vmax),
        (axs[0, 1], prior_mag, "Prior |u|", "viridis", mag_vmin, mag_vmax),
        (axs[0, 2], post_mag, "Posterior |u|", "viridis", mag_vmin, mag_vmax),
        (axs[1, 0], post_mag - prior_mag, "Posterior - Prior |u|", "coolwarm", -pp_lim, pp_lim),
        (axs[1, 1], prior_err, "Prior - CFD |u|", "coolwarm", -err_lim, err_lim),
        (axs[1, 2], post_err, "Posterior - CFD |u|", "coolwarm", -err_lim, err_lim),
    ]

    for ax, field, title, cmap, lo, hi in plot_items:
        pc = ax.contourf(Xs, Ys, field, levels=30, cmap=cmap, vmin=lo, vmax=hi)
        fig.colorbar(pc, ax=ax)
        if near_slice.any():
            ax.scatter(verts[near_slice, 0], verts[near_slice, 1], c="black", s=8)
        ax.set_aspect("equal")
        ax.set_title(f"{title}, z={z_here:.4g}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    return fig


def plot_pressure_slice(result, z_slice_target=2.5):
    """Posterior pressure field at a z-slice (only if pressure was fitted)."""
    if result.get("pressure_posterior") is None:
        return None

    res = result["res"]
    bounds = result["bounds"]
    test_points = result["test_points"]
    verts = result["mesh_vertices"]

    P = test_points.reshape(res, res, res, 3)
    press = result["pressure_posterior"].reshape(res, res, res)

    z_grid = np.linspace(bounds[2, 0], bounds[2, 1], res)
    k = int(np.argmin(np.abs(z_grid - z_slice_target)))
    z_here = z_grid[k]

    Xs = P[:, :, k, 0]
    Ys = P[:, :, k, 1]
    p_slice = press[:, :, k]

    slice_tol = 0.5 * abs(z_grid[1] - z_grid[0]) if res > 1 else 1e-6
    near_slice = np.abs(verts[:, 2] - z_here) <= slice_tol

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    pc = ax.contourf(Xs, Ys, p_slice, levels=30, cmap="viridis")
    fig.colorbar(pc, ax=ax, label="pressure")
    if near_slice.any():
        ax.scatter(verts[near_slice, 0], verts[near_slice, 1], c="black", s=8)
    ax.set_aspect("equal")
    ax.set_title(f"Posterior pressure, z={z_here:.4g}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig


def plot_all(result, z_slice_target=2.5, show=True):
    """Convenience: produce every figure."""
    figs = [
        plot_posterior_3d(result),
        plot_slice_comparison(result, z_slice_target),
    ]
    pf = plot_pressure_slice(result, z_slice_target)
    if pf is not None:
        figs.append(pf)
    if show:
        plt.show()
    return figs
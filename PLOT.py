import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

plt.style.use("dark_background")
mplstyle.use("fast")


def _velocity_magnitude(U):
    return np.sqrt(np.sum(U ** 2, axis=-1))


def _safe_levels(vmin: float, vmax: float, n: int = 30) -> np.ndarray:
    """Return strictly-increasing levels between vmin and vmax.

    Falls back to a two-element array [vmin, vmin+eps] when the field is
    constant (vmin == vmax), which would otherwise cause matplotlib to raise
    'contour levels must be increasing'.
    """
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if np.isclose(vmin, vmax):
        eps = max(abs(vmin) * 1e-6, 1e-10)
        return np.array([vmin - eps, vmax + eps])
    return np.linspace(vmin, vmax, n)


def _safe_contourf(ax, X, Y, Z, levels, **kwargs):
    """Call contourf, falling back to imshow on a flat/degenerate field."""
    try:
        return ax.contourf(X, Y, Z, levels=levels, **kwargs)
    except ValueError:
        # Field is effectively constant — render as a uniform colour patch.
        vmin, vmax = levels[0], levels[-1]
        cmap = kwargs.get("cmap", "viridis")
        return ax.contourf(X, Y, Z, levels=2, vmin=vmin, vmax=vmax, cmap=cmap,
                           extend=kwargs.get("extend", "both"))


def plot_posterior_3d(result, max_field_points=6000, field_alpha=0.06):
    """3D scatter of posterior velocity magnitude.

    Lag fixes:
      - Subsample the dense test-point cloud (the main bottleneck).
      - Drop the per-vertex black mesh cloud (huge, occludes everything).
      - Render the field cloud with very low alpha so the red training
        points stay readable, and draw them LAST with depthshade off.
    """
    test_points = np.asarray(result["test_points"])
    GPR_posterior = np.asarray(result["GPR_posterior"])
    training_coords = np.asarray(result["training_coords"])
    mesh_vertices = np.asarray(result['mesh_vertices'])
    v_inf = result['V_inf']

    vel_mags = _velocity_magnitude(GPR_posterior)

    # ---- subsample the background field for responsiveness ----
    n = test_points.shape[0]
    if n > max_field_points:
        idx = np.random.default_rng(0).choice(n, max_field_points, replace=False)
        tp = test_points[idx]
        vm = vel_mags[idx]
    else:
        tp = test_points
        vm = vel_mags

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")

    # faint translucent field so reds pop through
    sc = ax.scatter3D(tp[:, 0], tp[:, 1], tp[:, 2], c=vm, cmap="viridis", alpha=field_alpha, s=3, edgecolors="none", rasterized=True)
    fig.colorbar(sc, ax=ax, label="|velocity|")

    ax.scatter3D(mesh_vertices[:,0], mesh_vertices[:,1], mesh_vertices[:,2], c='black')
    ax.quiver(*(v_inf*3), *v_inf, length=3)

    # training samples drawn last, opaque, large, no depth fading
    ax.scatter3D(training_coords[:, 0], training_coords[:, 1], training_coords[:, 2], c="red", s=60, depthshade=False, edgecolors="white", linewidths=0.6, label="training samples", zorder=10)
    ax.legend(loc="upper right")

    ax.set_title("GPR posterior velocity magnitude")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))
    return fig


def _slice_index(z_grid, target):
    return int(np.argmin(np.abs(z_grid - target)))


def plot_slice_comparison(result, z_slice_target=2.5):
    """2x3 panel of CFD / prior / posterior |u| and their differences at a z-slice."""
    res = result["res"]
    bounds = result["bounds"]
    test_points = np.asarray(result["test_points"])
    verts = np.asarray(result["mesh_vertices"])

    P = test_points.reshape(res, res, res, 3)
    prior_U = np.asarray(result["means_tests"]).reshape(res, res, res, 3)
    post_U = np.asarray(result["GPR_posterior"]).reshape(res, res, res, 3)
    cfd_U = np.asarray(result["cfd_test_vels"]).reshape(res, res, res, 3)

    z_grid = np.linspace(bounds[2, 0], bounds[2, 1], res)
    k = _slice_index(z_grid, z_slice_target)
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

    mag_vmin = float(np.nanmin([cfd_mag, prior_mag, post_mag]))
    mag_vmax = float(np.nanmax([cfd_mag, prior_mag, post_mag]))
    err_lim = float(np.nanmax(np.abs([prior_err, post_err])))
    pp_lim = float(np.nanmax(np.abs(post_mag - prior_mag)))

    mag_levels = _safe_levels(mag_vmin, mag_vmax)
    err_levels = _safe_levels(-err_lim, err_lim)
    pp_levels  = _safe_levels(-pp_lim,  pp_lim)

    plot_items = [
        (axs[0, 0], cfd_mag, "CFD truth |u|", "viridis", mag_levels),
        (axs[0, 1], prior_mag, "Prior |u|", "viridis", mag_levels),
        (axs[0, 2], post_mag, "Posterior |u|", "viridis", mag_levels),
        (axs[1, 0], post_mag - prior_mag, "Posterior - Prior |u|", "coolwarm", pp_levels),
        (axs[1, 1], prior_err, "Prior - CFD |u|", "coolwarm", err_levels),
        (axs[1, 2], post_err, "Posterior - CFD |u|", "coolwarm", err_levels),
    ]

    for ax, field, title, cmap, levels in plot_items:
        pc = _safe_contourf(ax, Xs, Ys, field, levels=levels, cmap=cmap, extend="both")
        fig.colorbar(pc, ax=ax)
        if near_slice.any():
            ax.scatter(verts[near_slice, 0], verts[near_slice, 1], c="black", s=8)
        ax.set_aspect("equal")
        ax.set_title(f"{title}, z={z_here:.4g}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    return fig


def plot_multi_slices(result, n_slices=5, field="posterior", axis="z",
                      slice_range=(0.1, 6.5)):
    """Grid of |u| slices across several planes along the chosen axis.

    field:       "posterior", "variances", "prior", or "cfd"
    axis:        "x", "y", or "z"
    slice_range: (lo, hi) coordinate range along `axis` to sample slices
                 within; None spans the full bounds.
    """
    res = result["res"]
    bounds = result["bounds"]
    test_points = np.asarray(result["test_points"])

    P = test_points.reshape(res, res, res, 3)
    field_map = {
        "posterior": "GPR_posterior",
        "variances": "GPR_variances",
        "prior": "means_tests",
        "cfd": "cfd_test_vels",
    }
    U = np.asarray(result[field_map[field]]).reshape(res, res, res, 3)
    mag = _velocity_magnitude(U)

    ax_idx = {"x": 0, "y": 1, "z": 2}[axis]
    grid = np.linspace(bounds[ax_idx, 0], bounds[ax_idx, 1], res)

    if slice_range is None:
        lo, hi = grid[0], grid[-1]
    else:
        lo, hi = slice_range
    targets = np.linspace(lo, hi, n_slices)
    ks = np.unique([_slice_index(grid, t) for t in targets])

    vmin, vmax = float(np.nanmin(mag)), float(np.nanmax(mag))
    levels = _safe_levels(vmin, vmax)

    ncols = min(len(ks), 5)
    nrows = int(np.ceil(len(ks) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows),
                            constrained_layout=True, squeeze=False)

    other = [a for a in (0, 1, 2) if a != ax_idx]
    labels = ["x", "y", "z"]

    pc = None
    for i, k in enumerate(ks):
        ax = axs[i // ncols][i % ncols]
        if axis == "z":
            A, B, fld = P[:, :, k, other[0]], P[:, :, k, other[1]], mag[:, :, k]
        elif axis == "y":
            A, B, fld = P[:, k, :, other[0]], P[:, k, :, other[1]], mag[:, k, :]
        else:
            A, B, fld = P[k, :, :, other[0]], P[k, :, :, other[1]], mag[k, :, :]
        cmap = 'Reds' if field == 'variances' else 'viridis'
        pc = _safe_contourf(ax, A, B, fld, levels=levels, cmap=cmap, extend="both")
        ax.set_aspect("equal")
        ax.set_title(f"{axis}={grid[k]:.3g}")
        ax.set_xlabel(labels[other[0]])
        ax.set_ylabel(labels[other[1]])

    for j in range(len(ks), nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")

    if pc is not None:
        fig.colorbar(pc, ax=axs, label=f"|u| ({field})", shrink=0.8)
    fig.suptitle(f"{field} |u| slices along {axis}")
    return fig


def plot_pressure_slice(result, z_slice_target=2.5):
    """Posterior pressure field at a z-slice (only if pressure was fitted)."""
    if result.get("pressure_posterior") is None:
        return None

    res = result["res"]
    bounds = result["bounds"]
    test_points = np.asarray(result["test_points"])
    verts = np.asarray(result["mesh_vertices"])

    P = test_points.reshape(res, res, res, 3)
    press = np.asarray(result["pressure_posterior"]).reshape(res, res, res)

    z_grid = np.linspace(bounds[2, 0], bounds[2, 1], res)
    k = _slice_index(z_grid, z_slice_target)
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


def plot_all(result, z_slice_target=2.5, n_slices=5, show=True):
    figs = [
        plot_posterior_3d(result),
        plot_slice_comparison(result, z_slice_target),
        plot_multi_slices(result, n_slices=n_slices, field="posterior", axis="z", slice_range=(5,50)),
        plot_multi_slices(result, n_slices=n_slices, field="variances", axis="z", slice_range=(5,50))
        # plot_multi_slices(result, n_slices=n_slices, field="posterior", axis="z"),
        # plot_multi_slices(result, n_slices=n_slices, field="variances", axis="z")
    ]
    pf = plot_pressure_slice(result, z_slice_target)
    if pf is not None:
        figs.append(pf)
    if show:
        plt.show()
    return figs


def save_all(result, out_dir="plots", phase_label="phase0",
             z_slice_target=2.5, n_slices=5, show=False, close_after=True):
    """Build the standard figure set and save them to one multi-page PDF per phase.

    Each call writes  {out_dir}/{phase_label}.pdf  with one figure per page, so
    phases can be flipped through and compared side by side. The figures are the
    SAME ones plot_all draws - this just routes them to a file instead of (or as
    well as) the screen. Works regardless of how many training points the phase
    used, because every plotted FIELD is still res^3; only the scattered training
    coords change, and those are never reshaped.

    Returns the path to the written PDF.
    """
    import os
    from matplotlib.backends.backend_pdf import PdfPages

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{phase_label}.pdf")

    figs = [
        plot_posterior_3d(result),
        plot_slice_comparison(result, z_slice_target),
        plot_multi_slices(result, n_slices=n_slices, field="posterior", axis="z", slice_range=(5, 50)),
        plot_multi_slices(result, n_slices=n_slices, field="variances", axis="z", slice_range=(5, 50)),
    ]
    pf = plot_pressure_slice(result, z_slice_target)
    if pf is not None:
        figs.append(pf)

    # ---- build the 3-RMSE annotation text ----
    n_train = len(np.asarray(result["training_coords"]))
    m = result.get("metrics", {})

    def _fmt(v):
        return f"{v:.4g}" if isinstance(v, (int, float)) and v is not None else "n/a"

    dom = m.get("post_test_rmse"); dom_rel = m.get("rel_post_test_rmse")
    shl = m.get("post_shell_rmse"); shl_rel = m.get("rel_post_shell_rmse")
    fac = m.get("post_face_rmse"); fac_rel = m.get("rel_post_face_rmse")
    pres = m.get("pressure_test_rmse")

    header = f"{phase_label}  |  {n_train} training pts"
    rmse_lines = [
        f"velocity RMSE   1) whole domain: {_fmt(dom)} (rel {_fmt(dom_rel)})   "
        f"2) thick cylinder: {_fmt(shl)} (rel {_fmt(shl_rel)})   "
        f"3) on cylinder: {_fmt(fac)} (rel {_fmt(fac_rel)})",
        f"pressure RMSE: {_fmt(pres)}",
    ]
    annot = header + "\n" + "\n".join(rmse_lines)

    # corner annotation on every page
    for f in figs:
        f.text(0.01, 0.99, annot, ha="left", va="top", fontsize=8,
               color="0.8", family="monospace")

    # dedicated summary page at the FRONT of the PDF
    summary = plt.figure(figsize=(11, 8.5))
    summary.patch.set_facecolor("black")
    ax = summary.add_subplot(111); ax.axis("off")
    lines = [
        header, "",
        "Velocity RMSE (absolute  |  relative to truth-RMS):", "",
        f"  1. whole domain    : {_fmt(dom)}    (rel {_fmt(dom_rel)})    [{m.get('valid_cfd','?')} cells]",
        f"  2. thick cylinder  : {_fmt(shl)}    (rel {_fmt(shl_rel)})    [{m.get('shell_n','?')} pts]",
        f"  3. on the cylinder : {_fmt(fac)}    (rel {_fmt(fac_rel)})    [{m.get('face_n','?')} pts]",
        "", f"  Pressure RMSE      : {_fmt(pres)}",
        "", "", "Prior (potential-flow baseline) for reference:",
        f"  whole domain prior : {_fmt(m.get('prior_test_rmse'))}",
        f"  thick cyl prior    : {_fmt(m.get('prior_shell_rmse'))}",
        f"  on cylinder prior  : {_fmt(m.get('prior_face_rmse'))}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), ha="left", va="top",
            fontsize=13, color="white", family="monospace",
            transform=ax.transAxes)
    figs = [summary] + figs

    with PdfPages(pdf_path) as pdf:
        for f in figs:
            pdf.savefig(f, facecolor=f.get_facecolor())
    print(f"saved {len(figs)}-page figure set -> {pdf_path}")

    if show:
        plt.show()
    if close_after:
        for f in figs:
            plt.close(f)
    return pdf_path
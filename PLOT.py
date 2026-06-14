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


def triptych_field_vlim(results, z_slice_target=2.5, component=None):
    """Common scales for plot_slice_triptych across several results, so phases
    plotted separately share one colour scale.

    Returns (field_vlim, diff_vlim):
      field_vlim = (vmin, vmax) for the CFD/posterior panels (spans CFD and
                   posterior of every result; magnitude -> min/max, signed
                   component -> symmetric about zero).
      diff_vlim  = symmetric (-d, d) for the difference panel (max |posterior -
                   CFD| over every result). Pass BOTH to plot_slice_triptych for
                   each phase to fix every panel's scale (e.g. take them from
                   phase 0 and reuse on later phases).
    """
    if isinstance(results, dict):
        results = [results]
    lo, hi, lim, dlim = np.inf, -np.inf, 0.0, 0.0
    for result in results:
        res = result["res"]
        bounds = result["bounds"]
        z_grid = np.linspace(bounds[2, 0], bounds[2, 1], res)
        k = _slice_index(z_grid, z_slice_target)
        post_U = np.asarray(result["GPR_posterior"]).reshape(res, res, res, 3)
        cfd_U = np.asarray(result["cfd_test_vels"]).reshape(res, res, res, 3)

        def _f(U):
            return _velocity_magnitude(U[:, :, k, :]) if component is None \
                else U[:, :, k, component]

        cfd_f, post_f = _f(cfd_U), _f(post_U)
        for f in (cfd_f, post_f):
            lo = min(lo, float(np.nanmin(f)))
            hi = max(hi, float(np.nanmax(f)))
            lim = max(lim, float(np.nanmax(np.abs(f))))
        d = np.abs((post_f - cfd_f))
        d = d[np.isfinite(d)]
        if d.size:
            dlim = max(dlim, float(np.nanmax(d)))

    field_vlim = (lo, hi) if component is None else (-lim, lim)
    diff_vlim = (-dlim, dlim) if dlim > 0 else (-1.0, 1.0)
    return field_vlim, diff_vlim


def plot_variance_across_phases(results, z_slice_target=2.5, vlim=None,
                                phase_labels=None, show_cylinder=True):
    """One row of variance |u| slices at a fixed z, one panel per phase, all on
    a single shared colorbar so the phases are directly comparable.

    results:        list of per-phase result dicts (each needs the res^3
                    GPR_variances, i.e. a grid_eval run).
    z_slice_target: target z (m); nearest grid plane is used.
    vlim:           shared (vmin, vmax). None -> spans the variance of every
                    phase at this slice (auto shared scale).
    phase_labels:   optional list of titles; defaults to "Phase 0", "Phase 1"...
    show_cylinder:  overlay the integration ring on each panel.

    Note: absolute variance is not strictly comparable across phases because the
    kernel is refit each phase (shorter length scales in later phases raise the
    variance away from samples). The shared scale makes the SPATIAL pattern
    comparable; read it as a coverage diagnostic, not a convergence metric.
    """
    if isinstance(results, dict):
        results = [results]
    n = len(results)
    if phase_labels is None:
        phase_labels = [f"Phase {i}" for i in range(n)]

    # gather each phase's variance magnitude at the slice
    slices, Xs_list, Ys_list, z_used = [], [], [], None
    for result in results:
        res = result["res"]
        bounds = np.asarray(result["bounds"])
        P = np.asarray(result["test_points"]).reshape(res, res, res, 3)
        V = np.asarray(result["GPR_variances"]).reshape(res, res, res, 3)
        z_grid = np.linspace(bounds[2, 0], bounds[2, 1], res)
        k = _slice_index(z_grid, z_slice_target)
        z_used = z_grid[k]
        slices.append(_velocity_magnitude(V[:, :, k, :]))
        Xs_list.append(P[:, :, k, 0])
        Ys_list.append(P[:, :, k, 1])

    # shared scale across all phases at this slice
    if vlim is not None:
        vmin, vmax = float(vlim[0]), float(vlim[1])
    else:
        vmin = float(min(np.nanmin(s) for s in slices))
        vmax = float(max(np.nanmax(s) for s in slices))
    levels = _safe_levels(vmin, vmax)

    fig, axs = plt.subplots(1, n, figsize=(3.4 * n + 1.2, 4.4),
                            constrained_layout=True)
    if n == 1:
        axs = [axs]

    pc = None
    for ax, field, Xs, Ys, label, result in zip(
            axs, slices, Xs_list, Ys_list, phase_labels, results):
        pc = _safe_contourf(ax, Xs, Ys, field, levels=levels,
                            cmap="Reds", extend="max")
        if show_cylinder:
            _draw_ring(ax, _cylinder_ring_at_z(result, z_used))
        ax.set_aspect("equal")
        ax.set_title(label)
        ax.set_xlabel("x")
        if ax is axs[0]:
            ax.set_ylabel("y")

    fig.colorbar(pc, ax=axs, fraction=0.025, pad=0.01,
                 label=r"$\hat{\sigma}\,|u|$")
    fig.suptitle(f"Posterior variance across phases, z={z_used:.4g}")
    return fig


def multi_slice_vlim(results, field="variances", axis="z", slice_range=None):
    """Common (vmin, vmax) for plot_multi_slices across several results, so the
    same field (e.g. "variances") shares one colour scale across phases.

    Spans the whole field magnitude of every result (the slice_range/axis only
    pick which planes are drawn, not the data range, so the scale is taken over
    the full field). Pass the returned tuple as vlim to plot_multi_slices.
    """
    if isinstance(results, dict):
        results = [results]
    field_map = {"posterior": "GPR_posterior", "variances": "GPR_variances",
                 "prior": "means_tests", "cfd": "cfd_test_vels"}
    key = field_map[field]
    lo, hi = np.inf, -np.inf
    for result in results:
        res = result["res"]
        U = np.asarray(result[key]).reshape(res, res, res, 3)
        mag = _velocity_magnitude(U)
        lo = min(lo, float(np.nanmin(mag)))
        hi = max(hi, float(np.nanmax(mag)))
    return (lo, hi)


def _cylinder_ring_at_z(result, z_here):
    """(cx, cy, R) of the tilted cylinder where it cuts plane z=z_here, or None.
    The cylinder leans downstream with height, so the ring centre interpolates
    linearly from bottom_center to top_center while R is constant."""
    cg = result.get("cylinder_geom")
    if cg is None or cg.get("R") is None:
        return None
    bc = np.asarray(cg["bottom_center"], float)
    tc = np.asarray(cg["top_center"], float)
    dz = tc[2] - bc[2]
    frac = (z_here - bc[2]) / dz if abs(dz) > 1e-9 else 0.0
    cx = bc[0] + frac * (tc[0] - bc[0])
    cy = bc[1] + frac * (tc[1] - bc[1])
    return (cx, cy, float(cg["R"]))


def _draw_ring(ax, ring):
    """Overlay the cylinder integration ring (dashed white) on an axis."""
    if ring is not None:
        ax.add_patch(plt.Circle((ring[0], ring[1]), ring[2], fill=False,
                                edgecolor="white", linestyle="--", linewidth=1.4))


def plot_slice_triptych(result, z_slice_target=2.5, component=None,
                        field_vlim=None, diff_vlim=None, show_cylinder=True):
    """1x3 panel: CFD / posterior / (posterior - CFD) |u| at a z-slice.

    A trimmed sibling of plot_slice_comparison (prior field dropped) for the
    results chapter. Same house style: viridis magnitude, coolwarm difference,
    _safe_levels / _safe_contourf, black in-slice mesh scatter.

    component:  None -> velocity magnitude |u| (default); 0/1/2 -> signed u/v/w.
                For a signed component the field panels switch to coolwarm and a
                symmetric scale (a signed field has no natural zero floor).
    field_vlim: optional (vmin, vmax) for the CFD/posterior panels. Pass the same
                tuple to several phases (see triptych_field_vlim) so they share
                one field scale. None -> auto from this result alone.
    diff_vlim:  optional (-d, d) for the difference panel. Pass phase 0's value
                (from triptych_field_vlim) to every phase so the difference scale
                is fixed across phases too. None -> auto per phase (symmetric
                max-abs), which keeps small later-phase errors filling the range.
    """
    res = result["res"]
    bounds = result["bounds"]
    test_points = np.asarray(result["test_points"])
    verts = np.asarray(result["mesh_vertices"])

    P = test_points.reshape(res, res, res, 3)
    post_U = np.asarray(result["GPR_posterior"]).reshape(res, res, res, 3)
    cfd_U = np.asarray(result["cfd_test_vels"]).reshape(res, res, res, 3)

    z_grid = np.linspace(bounds[2, 0], bounds[2, 1], res)
    k = _slice_index(z_grid, z_slice_target)
    z_here = z_grid[k]

    Xs = P[:, :, k, 0]
    Ys = P[:, :, k, 1]

    def _field(U):
        return _velocity_magnitude(U[:, :, k, :]) if component is None \
            else U[:, :, k, component]

    cfd_f = _field(cfd_U)
    post_f = _field(post_U)
    diff = post_f - cfd_f

    slice_tol = 0.5 * abs(z_grid[1] - z_grid[0]) if res > 1 else 1e-6
    near_slice = np.abs(verts[:, 2] - z_here) <= slice_tol

    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)

    # shared field scale (matches plot_slice_comparison: span both fields), or
    # the caller-supplied common scale for cross-phase comparison.
    field_cmap = "viridis" if component is None else "coolwarm"
    if field_vlim is not None:
        f_vmin, f_vmax = float(field_vlim[0]), float(field_vlim[1])
    elif component is None:                     # magnitude -> viridis
        f_vmin = float(np.nanmin([cfd_f, post_f]))
        f_vmax = float(np.nanmax([cfd_f, post_f]))
    else:                                       # signed component -> symmetric
        f_lim = float(np.nanmax(np.abs([cfd_f, post_f])))
        f_vmin, f_vmax = -f_lim, f_lim
    field_levels = _safe_levels(f_vmin, f_vmax)

    # difference scale: caller-supplied shared limit, else auto symmetric max-abs.
    if diff_vlim is not None:
        d_lim = float(max(abs(diff_vlim[0]), abs(diff_vlim[1])))
    else:
        abs_diff = np.abs(diff[np.isfinite(diff)])
        d_lim = float(np.nanmax(abs_diff)) if abs_diff.size else 1.0
    if d_lim <= 0:
        d_lim = 1.0
    diff_levels = _safe_levels(-d_lim, d_lim)

    clab = "|u|" if component is None else ["u", "v", "w"][component]
    plot_items = [
        (axs[0], cfd_f, f"CFD truth {clab}", field_cmap, field_levels),
        (axs[1], post_f, f"Posterior {clab}", field_cmap, field_levels),
        (axs[2], diff, f"Posterior - CFD {clab}", "coolwarm", diff_levels),
    ]

    # Cylinder integration surface at this slice (where low error matters for
    # the force). The tilted cylinder leans downstream with height.
    ring = _cylinder_ring_at_z(result, z_here) if show_cylinder else None

    for ax, field, title, cmap, levels in plot_items:
        pc = _safe_contourf(ax, Xs, Ys, field, levels=levels, cmap=cmap, extend="both")
        fig.colorbar(pc, ax=ax)
        if near_slice.any():
            ax.scatter(verts[near_slice, 0], verts[near_slice, 1], c="black", s=8)
        if ax is axs[2]:                      # ring only on the difference panel
            _draw_ring(ax, ring)
        ax.set_aspect("equal")
        ax.set_title(f"{title}, z={z_here:.4g}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    return fig


def plot_multi_slices(result, n_slices=5, field="posterior", axis="z",
                      slice_range=(0.1, 6.5), vlim=None):
    """Grid of |u| slices across several planes along the chosen axis.

    field:       "posterior", "variances", "prior", or "cfd"
    axis:        "x", "y", or "z"
    slice_range: (lo, hi) coordinate range along `axis` to sample slices
                 within; None spans the full bounds.
    vlim:        optional (vmin, vmax) colour scale. Pass the same tuple across
                 phases (e.g. phase 0's, see multi_slice_vlim) to fix the scale.
                 None -> auto from this result alone.
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

    if vlim is not None:
        vmin, vmax = float(vlim[0]), float(vlim[1])
    else:
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


def plot_pressure_triptych(result, z_slice_target=2.5, field_vlim=None,
                           diff_vlim=None, show_cylinder=True):
    """1x3 panel: CFD / posterior / (posterior - CFD) pressure at a z-slice.

    The pressure analogue of plot_slice_triptych. Needs both the posterior
    pressure (result["pressure_posterior"]) and the CFD pressure truth on the
    grid (result["cfd_test_press"]); returns None if either is missing (e.g. a
    grid-free run, or fit_pressure=False).

    House style: the two field panels share one diverging scale centred on zero
    (pressure is signed) using coolwarm; the difference panel auto-scales to its
    own symmetric max-abs unless diff_vlim is given.

    field_vlim / diff_vlim: optional shared scales across phases (pass phase 0's,
    see pressure_triptych_vlim). None -> auto from this result alone.
    """
    if result.get("pressure_posterior") is None or result.get("cfd_test_press") is None:
        return None

    res = result["res"]
    bounds = result["bounds"]
    test_points = np.asarray(result["test_points"])
    verts = np.asarray(result["mesh_vertices"])

    P = test_points.reshape(res, res, res, 3)
    post_p = np.asarray(result["pressure_posterior"]).reshape(res, res, res)
    cfd_p = np.asarray(result["cfd_test_press"]).reshape(res, res, res)

    z_grid = np.linspace(bounds[2, 0], bounds[2, 1], res)
    k = _slice_index(z_grid, z_slice_target)
    z_here = z_grid[k]

    Xs = P[:, :, k, 0]
    Ys = P[:, :, k, 1]
    cfd_f = cfd_p[:, :, k]
    post_f = post_p[:, :, k]
    diff = post_f - cfd_f

    slice_tol = 0.5 * abs(z_grid[1] - z_grid[0]) if res > 1 else 1e-6
    near_slice = np.abs(verts[:, 2] - z_here) <= slice_tol

    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)

    # shared field scale: symmetric diverging (pressure is signed)
    if field_vlim is not None:
        f_vmin, f_vmax = float(field_vlim[0]), float(field_vlim[1])
    else:
        f_lim = float(np.nanmax(np.abs([cfd_f, post_f])))
        f_vmin, f_vmax = -f_lim, f_lim
    field_levels = _safe_levels(f_vmin, f_vmax)

    # difference scale: caller-supplied shared limit, else auto symmetric max-abs
    if diff_vlim is not None:
        d_lim = float(max(abs(diff_vlim[0]), abs(diff_vlim[1])))
    else:
        abs_diff = np.abs(diff[np.isfinite(diff)])
        d_lim = float(np.nanmax(abs_diff)) if abs_diff.size else 1.0
    if d_lim <= 0:
        d_lim = 1.0
    diff_levels = _safe_levels(-d_lim, d_lim)

    plot_items = [
        (axs[0], cfd_f, "CFD truth p", "coolwarm", field_levels),
        (axs[1], post_f, "Posterior p", "coolwarm", field_levels),
        (axs[2], diff, "Posterior - CFD p", "coolwarm", diff_levels),
    ]

    ring = _cylinder_ring_at_z(result, z_here) if show_cylinder else None

    for ax, field, title, cmap, levels in plot_items:
        pc = _safe_contourf(ax, Xs, Ys, field, levels=levels, cmap=cmap, extend="both")
        fig.colorbar(pc, ax=ax)
        if near_slice.any():
            ax.scatter(verts[near_slice, 0], verts[near_slice, 1], c="black", s=8)
        if ax is axs[2]:                      # ring only on the difference panel
            _draw_ring(ax, ring)
        ax.set_aspect("equal")
        ax.set_title(f"{title}, z={z_here:.4g}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    return fig


def pressure_triptych_vlim(results, z_slice_target=2.5):
    """Common (field_vlim, diff_vlim) for plot_pressure_triptych across phases.
    Mirrors triptych_field_vlim but for the scalar pressure field. Returns
    (None, None) if any result lacks pressure fields."""
    if isinstance(results, dict):
        results = [results]
    flim, dlim = 0.0, 0.0
    for result in results:
        if result.get("pressure_posterior") is None or result.get("cfd_test_press") is None:
            return (None, None)
        res = result["res"]
        bounds = result["bounds"]
        z_grid = np.linspace(bounds[2, 0], bounds[2, 1], res)
        k = _slice_index(z_grid, z_slice_target)
        post_f = np.asarray(result["pressure_posterior"]).reshape(res, res, res)[:, :, k]
        cfd_f = np.asarray(result["cfd_test_press"]).reshape(res, res, res)[:, :, k]
        flim = max(flim, float(np.nanmax(np.abs([cfd_f, post_f]))))
        d = np.abs(post_f - cfd_f)
        d = d[np.isfinite(d)]
        if d.size:
            dlim = max(dlim, float(np.nanmax(d)))
    field_vlim = (-flim, flim) if flim > 0 else (-1.0, 1.0)
    diff_vlim = (-dlim, dlim) if dlim > 0 else (-1.0, 1.0)
    return field_vlim, diff_vlim


def plot_all(result, z_slice_target=2.5, n_slices=5, show=True):
    figs = [
        plot_posterior_3d(result),
        plot_slice_comparison(result, z_slice_target),
        plot_slice_triptych(result, z_slice_target),
        plot_multi_slices(result, n_slices=n_slices, field="posterior", axis="z", slice_range=(1,75)),
        plot_multi_slices(result, n_slices=n_slices, field="variances", axis="z", slice_range=(1,75))
    ]
    pt = plot_pressure_triptych(result, z_slice_target)
    if pt is not None:
        figs.append(pt)
    pf = plot_pressure_slice(result, z_slice_target)
    if pf is not None:
        figs.append(pf)
    if show:
        plt.show()
    return figs


def save_all(result, out_dir="plots", phase_label="phase0",
             z_slice_target=2.5, n_slices=5, show=False, close_after=True,
             true_force=None, field_vlim=None, diff_vlim=None,
             var_vlim=None, press_vlim=None, press_diff_vlim=None):
    """Build the standard figure set and save them to one multi-page PDF per phase.

    Each call writes  {out_dir}/{phase_label}.pdf  with one figure per page, so
    phases can be flipped through and compared side by side. The figures are the
    SAME ones plot_all draws - this just routes them to a file instead of (or as
    well as) the screen. Works regardless of how many training points the phase
    used, because every plotted FIELD is still res^3; only the scattered training
    coords change, and those are never reshaped.

    Also writes clean per-figure PNGs (no annotation overlay, no summary page)
    into {out_dir}/png/ for direct inclusion in the report.

    field_vlim / diff_vlim:        shared scales for the velocity triptych.
    var_vlim:                      shared scale for the variance multi-slice.
    press_vlim / press_diff_vlim:  shared scales for the pressure triptych.
    Each is optional; pass phase 0's values (see triptych_field_vlim,
    multi_slice_vlim, pressure_triptych_vlim) to fix scales across phases.
    None -> auto per phase.

    Returns the path to the written PDF.
    """
    import os
    from matplotlib.backends.backend_pdf import PdfPages

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{phase_label}.pdf")

    figs = [
        plot_posterior_3d(result),
        plot_slice_comparison(result, z_slice_target),
        plot_slice_triptych(result, z_slice_target, field_vlim=field_vlim, diff_vlim=diff_vlim),
        plot_multi_slices(result, n_slices=n_slices, field="posterior", axis="z", slice_range=(1, 75)),
        plot_multi_slices(result, n_slices=n_slices, field="variances", axis="z", slice_range=(1, 75), vlim=var_vlim),
    ]
    pt = plot_pressure_triptych(result, z_slice_target,
                                field_vlim=press_vlim, diff_vlim=press_diff_vlim)
    if pt is not None:
        figs.append(pt)
    pf = plot_pressure_slice(result, z_slice_target)
    if pf is not None:
        figs.append(pf)

    # ---- clean PNGs FIRST, before any annotation is stamped on ----
    # These are the report figures: the bare matplotlib plots with no corner
    # text and no summary page. Page-numbered to match figs order.
    png_dir = os.path.join(out_dir, "png")
    os.makedirs(png_dir, exist_ok=True)
    for i, f in enumerate(figs, start=1):
        png_path = os.path.join(png_dir, f"{phase_label}_p{i:02d}.png")
        f.savefig(png_path, facecolor=f.get_facecolor(), dpi=200)
    print(f"saved {len(figs)} clean PNG(s) -> {png_dir}/{phase_label}_p*.png")

    # ---- build the 3-RMSE annotation text ----
    n_train = len(np.asarray(result["training_coords"]))
    m = result.get("metrics", {})

    def _fmt(v):
        return f"{v:.4g}" if isinstance(v, (int, float)) and v is not None else "n/a"

    dom = m.get("post_test_rmse"); dom_rel = m.get("rel_post_test_rmse")
    pres = m.get("pressure_test_rmse")
    fmag = m.get("force_mag"); fvec = m.get("force_vec")

    def _fvec(v):
        if v is None:
            return "n/a"
        return "[" + ", ".join(f"{c:.4g}" for c in v) + "]"

    header = f"{phase_label}  |  {n_train} training pts"
    rmse_lines = [
        f"velocity RMSE (whole domain): {_fmt(dom)} (rel {_fmt(dom_rel)})",
        f"pressure RMSE: {_fmt(pres)}    momentum force |F|: {_fmt(fmag)}",
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
        f"  whole domain       : {_fmt(dom)}    (rel {_fmt(dom_rel)})    [{m.get('valid_cfd','?')} cells]",
        "", f"  Pressure RMSE      : {_fmt(pres)}",
        "", "Momentum-integral force:",
        f"  |F|                : {_fmt(fmag)}    [{m.get('momentum_n','?')} surface pts]",
        f"  F vector           : {_fvec(fvec)}",
    ]
    if true_force is not None:
        import numpy as _np
        tf = _np.asarray(true_force, float)
        lines.append(f"  F true             : {_fvec(tf.tolist())}    |F_true|={_np.linalg.norm(tf):.6g}")
        if fvec is not None:
            err = _np.asarray(fvec, float) - tf
            rel = _np.linalg.norm(err) / max(_np.linalg.norm(tf), 1e-12)
            lines.append(f"  force error dF     : {_fvec(err.tolist())}    rel={rel:.4g}")
    lines += [
        "", "", "Prior (potential-flow baseline) for reference:",
        f"  whole domain prior : {_fmt(m.get('prior_test_rmse'))}",
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


def plot_force_convergence(force_curve, true_force=None, out_dir="plots",
                           filename="force_convergence.pdf", show=False):
    """Plot the three force components (Fx, Fy, Fz) across phases vs # drones.

    force_curve: list of (n_drones, force_mag, force_vec) tuples, one per phase.
    Each component gets its own panel with the true value drawn as a dashed line,
    and every point is annotated with its drone count.
    """
    import os
    rows = [(n, fm, fv) for (n, fm, fv) in force_curve if fv is not None]
    if not rows:
        print("plot_force_convergence: no force data to plot.")
        return None

    ns = np.array([r[0] for r in rows])
    F = np.array([r[2] for r in rows], dtype=float)   # (phases, 3)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    labels = ["Fx", "Fy", "Fz"]
    colors = ["#5DCAA5", "#85B7EB", "#EF9F27"]
    for k, (ax, lab, col) in enumerate(zip(axes, labels, colors)):
        ax.plot(ns, F[:, k], "-o", color=col, label=lab, linewidth=1.8, markersize=6)
        if true_force is not None:
            tv = float(np.asarray(true_force, float)[k])
            ax.axhline(tv, ls="--", color="0.6", linewidth=1.2,
                       label=f"true {lab}={tv:.4g}")
        # annotate drone counts at each point
        for xi, yi in zip(ns, F[:, k]):
            ax.annotate(f"{xi}", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color="0.75")
        ax.set_ylabel(f"{lab}  [N]")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("number of training drones")
    axes[0].set_title("Momentum force convergence vs # drones", fontsize=12)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, facecolor=fig.get_facecolor())
    print(f"saved force-convergence plot -> {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path
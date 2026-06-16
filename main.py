from GPR import run_gpr
from PLOT import (plot_all, save_all, plot_force_convergence,
                  triptych_field_vlim, multi_slice_vlim, pressure_triptych_vlim,
                  plot_variance_across_phases)
from adaptive import propose_adaptive_points, propose_top_cap_points
import numpy as np
import trimesh
from flowpanelwrapper import FLOWPanelSolver


# ---- shared run settings (kept identical across phases so only sampling changes) ----
COMMON = dict(
    stl_filepath="input_stls/Aerospecial_building4.stl",
    cfd_filepath="inputs/csv_with_everything.pkl",
    stl_scale=1.0 / 1000.0,
    res=100,
    v_inf=(0.0, 13.6, 0.0),
    bounds_input=np.array([[-100, 100], [30, 275], [0, 80]]),
    n_restarts=6,
    fit_pressure=True,
    posterior_batch=100,
    compute_variance=True,
    var_res=50,
    grid_eval=False,
)

# initial sampling: tilted cylinder (gives us cylinder_geom for the adaptive regions)
INITIAL_SAMPLING = dict(
    sample_method="cylinder",
    sample_config={"r_factor": 1.2, "h_factor": 1.5, "tilt_deg": 10,
                   "n_points": 100, "front_frac": 0.25, "front_half_angle_deg": 45,
                   "top_cap": True, "top_cap_frac": 0.2},
)

# how many adaptive phases to run (0 -> behaves exactly like the old single run)
N_PHASES = 3

# DEPRECATED, USE TOP_CAP IN INITIAL SAMPLING optional FINAL top-cap phase: places a few drones on the cylinder top cap,
# where momentum may escape the open control volume. On by default.
TOP_CAP_PHASE = False
TOP_CAP_N = 50            # number of cap drones
TOP_CAP_Z_OFFSET = 0.0   # place cap at z_top (of the CYLINDER) + this

# Known true force [Fx, Fy, Fz] in N, for convergence comparison each phase.
# Set to None to disable the comparison printout.
TRUE_FORCE = np.array([155433.0, 208647.0, 72586.0])

# per-phase adaptive config (overrides adaptive.ADAPTIVE_DEFAULTS)
ADAPTIVE_CFG = dict(
    # difficulty score (computed on the existing res^3 grid via np.gradient)
    w_var=0.2, w_grad=0.4, w_vort=0.4,   # favor gradient + vorticity
    # weighted-LHS candidate pool over the thick tilted cylinder shell
    pool_size=4000, resample_size=600, score_beta=4.0,
    shell_thick_in=0.20, shell_thick_out=0.20,   # shell spans 0.7R .. 1.3R
    front_frac=0.5, front_half_angle_deg=60.0,   # bias toward the wake side
    # per-phase budget (mostly on-cylinder)
    n_new=100, frac_region1=0.70, frac_region2=0.15, frac_region3=0.15,
    # spacing: hard drone limit + optional spread relaxation
    excl_horizontal=1.2, excl_vertical=4.2,
    spread_radius=None,   # set e.g. 6.0 to relax points ~6 m apart laterally
    fd_step=[1.0,1.0,1.0]
)

ADAPTIVE_CFG_PER_PHASE = {
    # 1: dict(w_grad=0.35, w_vort=0.35, w_var=0.3),
    # 2: dict(w_grad=0.4, w_vort=0.4, w_var=0.2),
    # 3: dict(w_grad=0.5, w_vort=0.5, w_var=0.0),
    # 4: dict(w_grad=0.34, w_vort=0.33, w_var=0.33),
    # 5: dict(w_grad=0.4, w_vort=0.4, w_var=0.2),
}


def _phase_cfg(phase):
    """Merge ADAPTIVE_CFG with any per-phase overrides for this phase."""
    cfg = dict(ADAPTIVE_CFG)
    cfg.update(ADAPTIVE_CFG_PER_PHASE.get(phase, {}))
    cfg["pool_seed"] = ADAPTIVE_CFG.get("seed", 7) + phase   # fresh pool each phase
    return cfg


def _print_rmse(phase, result):
    """Print the three velocity RMSEs (whole domain / thick cylinder / on-face),
    each as absolute and relative-to-truth-RMS, plus pressure."""
    m = result["metrics"]
    def fmt(v):
        return f"{v:.4g}" if v is not None else "n/a"
    dom = m.get("post_test_rmse")
    dom_rel = m.get("rel_post_test_rmse")
    print(f"\n[RMSE] phase {phase}  (training pts: {m.get('training_point_n')})")
    print(f"    1. whole domain   : {fmt(dom)}   (rel {fmt(dom_rel)})   "
          f"[{m.get('valid_cfd')} cells]")
    print(f"    pressure          : {fmt(m.get('pressure_test_rmse'))}")
    fmag = m.get("force_mag"); fvec = m.get("force_vec")
    if fmag is not None and fvec is not None:
        fx, fy, fz = (float(c) for c in fvec)
        print(f"    momentum force    : |F|={fmt(fmag)}   "
              f"[{m.get('momentum_n')} surface pts]")
        print(f"        Fx={fx:12.6g}   Fy={fy:12.6g}   Fz={fz:12.6g}")
        if TRUE_FORCE is not None:
            tf = np.asarray(TRUE_FORCE, float)
            err = np.asarray(fvec, float) - tf
            ex, ey, ez = (float(c) for c in err)
            rel = np.linalg.norm(err) / max(np.linalg.norm(tf), 1e-12)
            # per-direction relative error: |dF_i| / |true_i|
            def _relc(e, t):
                return abs(e) / abs(t) if abs(t) > 1e-9 else float("nan")
            rx, ry, rz = _relc(ex, tf[0]), _relc(ey, tf[1]), _relc(ez, tf[2])
            print(f"        true            "
                  f"Fx={tf[0]:12.6g}   Fy={tf[1]:12.6g}   Fz={tf[2]:12.6g}")
            print(f"        error dF        "
                  f"dFx={ex:+11.5g}  dFy={ey:+11.5g}  dFz={ez:+11.5g}   |dF|/|F_true|={rel:.4g}")
            print(f"        rel err / dir   "
                  f"x={rx:11.4g}  y={ry:11.4g}  z={rz:11.4g}")
    elif fmag is not None:
        print(f"    momentum force    : |F|={fmt(fmag)}   [{m.get('momentum_n')} surface pts]")

def count_front_back(accumulated, stl_mesh, v_inf, stl_scale=1.0):
    """Count accumulated drones upstream vs downstream of the building center,
    split by the plane through the center normal to the horizontal flow."""

    V = np.asarray(stl_mesh.vertices) * stl_scale   # match coords to training pts

    print(f"    [debug] V x-range: {V[:,0].min():.3g}..{V[:,0].max():.3g}  "
          f"P0={0.5*(V[:,0].min()+V[:,0].max()):.3g}, "
          f"{0.5*(V[:,1].min()+V[:,1].max()):.3g}")

    v = np.asarray(v_inf, float)
    s = np.array([v[0], v[1], 0.0])
    s = s / np.linalg.norm(s)                        # downstream (+wake) direction

    P0 = np.array([0.5 * (V[:, 0].min() + V[:, 0].max()),
                   0.5 * (V[:, 1].min() + V[:, 1].max())])

    pts = np.asarray(accumulated, float)
    proj = (pts[:, :2] - P0[None, :]) @ s[:2]        # signed streamwise distance

    eps = 1e-9
    n_front = int(np.sum(proj > eps))                # downstream
    n_back = int(np.sum(proj < -eps))               # upstream
    n_on = int(np.sum(np.abs(proj) <= eps))          # exactly on the plane

    print("\n=== front/back split of accumulated drones (along flow) ===")
    print(f"    total              : {len(pts)}")
    print(f"    front (downstream) : {n_front}  ({100*n_front/len(pts):.1f}%)")
    print(f"    behind (upstream)  : {n_back}  ({100*n_back/len(pts):.1f}%)")
    if n_on:
        print(f"    on dividing plane  : {n_on}")
    return n_front, n_back


def _variance_snapshot(result, cylinder_geom):
    """Light copy of just what plot_variance_across_phases needs from a phase,
    taken BEFORE _free_heavy strips the grid arrays. Keeps the full res^3
    variance + test grid (a few hundred MB total across phases at res=100, far
    smaller than the CFD interpolator that _free_heavy actually guards), so the
    cross-phase variance figure can be built after the loop."""
    return dict(
        res=result["res"],
        bounds=np.asarray(result["bounds"]),
        test_points=np.asarray(result["test_points"]),
        GPR_variances=np.asarray(result["GPR_variances"]),
        cylinder_geom=cylinder_geom,
    )


def main():
    # ---------- shared Julia panel solver (built ONCE, reused every phase) ----------
    # The panel solver depends only on (mesh, v_inf), which are constant across
    # all phases in a run. Previously each run_gpr launched + JIT-compiled +
    # tore down its own Julia process; building one here and passing it in skips
    # that per-phase cold start. Closed once in the finally block below.
    _shared_mesh = trimesh.load_mesh(COMMON["stl_filepath"])
    if COMMON["stl_scale"] != 1.0:
        _shared_mesh.apply_scale(COMMON["stl_scale"])
    shared_solver = FLOWPanelSolver(
        _shared_mesh, COMMON["v_inf"], julia_script="FP.jl",
        julia_bin="julia", verbose=False)
    try:
        return _main_impl(shared_solver)
    finally:
        try:
            shared_solver.close()
        except Exception:
            pass


def _main_impl(shared_solver):
    # ---------- phase 0: initial cylinder run ----------
    print("\n=== PHASE 0 (initial cylinder sampling) ===")
    result = run_gpr(**COMMON, **INITIAL_SAMPLING, solver=shared_solver)

    # The adaptive regions always attach to the ORIGINAL cylinder. Re-runs use
    # sample_method="array", which returns cylinder_geom=None, so we capture the
    # phase-0 geometry once and re-inject it into every later result.
    cylinder_geom = result.get("cylinder_geom")
    if cylinder_geom is None:
        raise RuntimeError(
            "Phase 0 must use sample_method='cylinder' - no cylinder_geom was produced."
        )

    # the accumulating set of all flown points (cylinder + every adaptive batch)
    accumulated = np.asarray(result["training_coords"], float)

    # Plotting needs the res^3 grid fields, which only exist when grid_eval=True.
    PLOTS = bool(COMMON.get("grid_eval", False))
    if PLOTS:
        FIELD_VLIM, DIFF_VLIM = triptych_field_vlim(result, z_slice_target=25)
        # VAR_VLIM = multi_slice_vlim(result, field="variances")
        VAR_VLIM = (0.0, 7.0)
        PRESS_VLIM, PRESS_DIFF_VLIM = pressure_triptych_vlim(result, z_slice_target=25)
        PRESS_DIFF_VLIM = (-30.0, 30.0)
    else:
        FIELD_VLIM = DIFF_VLIM = VAR_VLIM = PRESS_VLIM = PRESS_DIFF_VLIM = None

    # light per-phase variance snapshots for the cross-phase variance figure
    variance_snaps = []
    if PLOTS:
        save_all(result, out_dir="plots", phase_label="phase0", z_slice_target=25,
                 show=False, true_force=TRUE_FORCE,
                 field_vlim=FIELD_VLIM, diff_vlim=DIFF_VLIM, var_vlim=VAR_VLIM,
                 press_vlim=PRESS_VLIM, press_diff_vlim=PRESS_DIFF_VLIM)
        variance_snaps.append(_variance_snapshot(result, cylinder_geom))
    _print_rmse(0, result)
    results = [result]
    force_curve = [(len(accumulated), result["metrics"].get("force_mag"), result["metrics"].get("force_vec"))]

    def _free_heavy(res):
        """Drop the big arrays + the CFD sampler closure from an old phase result."""
        if res is None:
            return
        closer = res.get("_close_solver")
        if closer is not None:
            closer()
        for k in ("sample_dat_shi", "test_points", "GPR_posterior", "GPR_variances",
                  "means_tests", "cfd_test_vels", "pressure_posterior", "momentum",
                  "_prior_fn", "_close_solver", "_chol_c", "_chol_low"):
            res.pop(k, None)

    # ---------- adaptive phases ----------
    for phase in range(1, N_PHASES + 1):
        print(f"\n=== PHASE {phase} (adaptive) ===")
        # ensure the (array-based) latest result carries the original cylinder geom
        results[-1]["cylinder_geom"] = cylinder_geom
        new_pts = propose_adaptive_points(
            results[-1],
            previous_coords=accumulated,     # clear of ALL prior points
            adaptive_config=_phase_cfg(phase),
        )
        if len(new_pts) == 0:
            print("[main] no new points proposed - stopping early.")
            break

        accumulated = np.vstack([accumulated, new_pts])
        print(f"[main] accumulated training points: {len(accumulated)}")

        # free the previous phase's heavy data (CFD sampler, grid arrays) so the
        # next run_gpr's fresh 21M-point interpolator has room.
        _free_heavy(results[-1])

        # re-run keeping every previous point, just with the augmented set
        result = run_gpr(
            **COMMON,
            sample_method="array",
            samples=accumulated,             # method='array' takes explicit coords
            cylinder_geom_override=cylinder_geom,  # shell/face RMSE + momentum force here too
            solver=shared_solver,
        )
        result["cylinder_geom"] = cylinder_geom   # keep geometry available downstream
        if PLOTS:
            save_all(result, out_dir="plots", phase_label=f"phase{phase}", z_slice_target=25, show=False, true_force=TRUE_FORCE, field_vlim=FIELD_VLIM, diff_vlim=DIFF_VLIM, var_vlim=VAR_VLIM, press_vlim=PRESS_VLIM, press_diff_vlim=PRESS_DIFF_VLIM)
            # snapshot this phase's variance slice BEFORE it is freed next iteration
            variance_snaps.append(_variance_snapshot(result, cylinder_geom))
        results.append(result)
        _print_rmse(phase, result)
        force_curve.append((len(accumulated), result["metrics"].get("force_mag"), result["metrics"].get("force_vec")))

    # ---------- optional FINAL top-cap phase ----------
    if TOP_CAP_PHASE:
        print(f"\n=== FINAL PHASE (top cap) ===")
        results[-1]["cylinder_geom"] = cylinder_geom
        cap_pts = propose_top_cap_points(
            results[-1], previous_coords=accumulated,
            n_new=TOP_CAP_N, cap_z_offset=TOP_CAP_Z_OFFSET,
            adaptive_config=ADAPTIVE_CFG)
        if len(cap_pts):
            accumulated = np.vstack([accumulated, cap_pts])
            print(f"[main] accumulated training points: {len(accumulated)}")
            result = run_gpr(
                **COMMON, sample_method="array", samples=accumulated,
                cylinder_geom_override=cylinder_geom, solver=shared_solver)
            result["cylinder_geom"] = cylinder_geom
            if PLOTS:
                save_all(result, out_dir="plots", phase_label="phase_topcap",
                         z_slice_target=25, show=False, true_force=TRUE_FORCE,
                         field_vlim=FIELD_VLIM, diff_vlim=DIFF_VLIM, var_vlim=VAR_VLIM,
                         press_vlim=PRESS_VLIM, press_diff_vlim=PRESS_DIFF_VLIM)
            results.append(result)
            _print_rmse("top-cap", result)
            force_curve.append((len(accumulated), result["metrics"].get("force_mag"), result["metrics"].get("force_vec")))

    # ---------- force vs #drones summary ----------
    print("\n=== momentum force vs # training drones ===")
    if TRUE_FORCE is not None:
        tf = np.asarray(TRUE_FORCE, float)
        print(f"    {'drones':>6}  {'Fx':>12}  {'Fy':>12}  {'Fz':>12}  {'|F|':>12}  "
              f"{'relx':>8}  {'rely':>8}  {'relz':>8}  {'rel|F|':>8}")
        print(f"    {'TRUE':>6}  {tf[0]:12.6g}  {tf[1]:12.6g}  "
              f"{tf[2]:12.6g}  {np.linalg.norm(tf):12.6g}  "
              f"{'-':>8}  {'-':>8}  {'-':>8}  {'-':>8}")
    else:
        print(f"    {'drones':>6}  {'Fx':>12}  {'Fy':>12}  {'Fz':>12}  {'|F|':>12}")
    for entry in force_curve:
        n_pts, fmag, fvec = entry
        if fvec is None:
            print(f"    {n_pts:6d}  {'n/a':>12}")
            continue
        fx, fy, fz = (float(c) for c in fvec)
        row = f"    {n_pts:6d}  {fx:12.6g}  {fy:12.6g}  {fz:12.6g}  {fmag:12.6g}"
        if TRUE_FORCE is not None:
            err = np.asarray(fvec, float) - tf
            def _relc(e, t):
                return abs(e) / abs(t) if abs(t) > 1e-9 else float("nan")
            relF = np.linalg.norm(err) / max(np.linalg.norm(tf), 1e-12)
            row += (f"  {_relc(err[0], tf[0]):8.4g}  {_relc(err[1], tf[1]):8.4g}  "
                    f"{_relc(err[2], tf[2]):8.4g}  {relF:8.4g}")
        print(row)

    count_front_back(accumulated, results[-1]["stl_mesh"],
                     COMMON["v_inf"], stl_scale=1.0)
    # per-component force-convergence plot (annotated with drone counts)
    plot_force_convergence(force_curve, true_force=TRUE_FORCE, out_dir="plots")

    # ---------- cross-phase variance comparison (one row, shared scale) ----------
    if PLOTS and len(variance_snaps) > 1:
        fig = plot_variance_across_phases(variance_snaps, z_slice_target=25,
                                          vlim=VAR_VLIM)
        fig.savefig("plots/variance_across_phases.png",
                    facecolor=fig.get_facecolor(), dpi=200)
        print("saved cross-phase variance figure -> plots/variance_across_phases.png")

    # ---------- plot the final phase (only if the grid was evaluated) ----------
    if PLOTS:
        plot_all(results[-1], z_slice_target=25, show=True)
    return results


if __name__ == "__main__":
    main()
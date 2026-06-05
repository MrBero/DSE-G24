from GPR import run_gpr
from PLOT import plot_all, save_all, plot_force_convergence
from adaptive import propose_adaptive_points, propose_top_cap_points
import numpy as np


# ---- shared run settings (kept identical across phases so only sampling changes) ----
COMMON = dict(
    stl_filepath="input_stls/Aerospecial_building4.stl",
    cfd_filepath="inputs/csv_with_everything.pkl",
    stl_scale=1.0 / 1000.0,
    res=50,
    v_inf=(0.0, 13.6, 0.0),
    bounds_input=np.array([[-100, 100], [30, 275], [0, 50]]),
    n_restarts=6,
    fit_pressure=True,
    posterior_batch=100,
    compute_variance=True,
    var_res=50,          # variance computed on a 50^3 grid, interpolated up to res^3 (velocity stays full res)
    rmse_shell_frac=0.3, # in-cylinder RMSE shell: (1-f)R .. (1+f)R around the tilted rings
    rmse_face_halfwidth=0.05,  # on-face RMSE band: |rho - R| <= this*R
)

# initial sampling: tilted cylinder (gives us cylinder_geom for the adaptive regions)
INITIAL_SAMPLING = dict(
    sample_method="cylinder",
    sample_config={"r_factor": 1.2, "h_factor": 1.5, "tilt_deg": 10,
                   "n_points": 80, "front_frac": 0.25, "front_half_angle_deg": 45},
)

# how many adaptive phases to run (0 -> behaves exactly like the old single run)
N_PHASES = 3

# optional FINAL top-cap phase: places a few drones on the cylinder top cap,
# where momentum may escape the open control volume. On by default.
TOP_CAP_PHASE = False
TOP_CAP_N = 10            # number of cap drones
TOP_CAP_Z_OFFSET = 0.0   # place cap at z_top (of the CYLINDER) + this

# Known true force [Fx, Fy, Fz] in N, for convergence comparison each phase.
# Set to None to disable the comparison printout.
TRUE_FORCE = np.array([155433.0, 208647.0, 72586.0])

# per-phase adaptive config (overrides adaptive.ADAPTIVE_DEFAULTS)
ADAPTIVE_CFG = dict(
    # difficulty score (computed on the existing res^3 grid via np.gradient)
    w_var=0.2, w_grad=0.4, w_vort=0.4,   # favor gradient + vorticity
    # weighted-LHS candidate pool over the thick tilted cylinder shell
    pool_size=4000, resample_size=600, score_beta=2.0,
    shell_thick_in=0.40, shell_thick_out=0.40,   # shell spans 0.7R .. 1.3R
    front_frac=0.5, front_half_angle_deg=60.0,   # bias toward the wake side
    # per-phase budget (mostly on-cylinder)
    n_new=80, frac_region1=0.60, frac_region2=0.2, frac_region3=0.2,
    # spacing: hard drone limit + optional spread relaxation
    excl_horizontal=1.2, excl_vertical=4.2,
    spread_radius=None,   # set e.g. 6.0 to relax points ~6 m apart laterally
)


def _print_rmse(phase, result):
    """Print the three velocity RMSEs (whole domain / thick cylinder / on-face),
    each as absolute and relative-to-truth-RMS, plus pressure."""
    m = result["metrics"]
    def fmt(v):
        return f"{v:.4g}" if v is not None else "n/a"
    dom = m.get("post_test_rmse")
    dom_rel = m.get("rel_post_test_rmse")
    shell = m.get("post_shell_rmse"); shell_rel = m.get("rel_post_shell_rmse")
    face = m.get("post_face_rmse"); face_rel = m.get("rel_post_face_rmse")
    print(f"\n[RMSE] phase {phase}  (training pts: {m.get('training_point_n')})")
    print(f"    1. whole domain   : {fmt(dom)}   (rel {fmt(dom_rel)})   "
          f"[{m.get('valid_cfd')} cells]")
    print(f"    2. thick cylinder : {fmt(shell)}   (rel {fmt(shell_rel)})   "
          f"[{m.get('shell_n')} cells]")
    print(f"    3. on the cylinder: {fmt(face)}   (rel {fmt(face_rel)})   "
          f"[{m.get('face_n')} cells]")
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


def main():
    # ---------- phase 0: initial cylinder run ----------
    print("\n=== PHASE 0 (initial cylinder sampling) ===")
    result = run_gpr(**COMMON, **INITIAL_SAMPLING)

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

    save_all(result, out_dir="plots", phase_label="phase0", z_slice_target=25, show=False, true_force=TRUE_FORCE)
    _print_rmse(0, result)
    results = [result]
    force_curve = [(len(accumulated), result["metrics"].get("force_mag"), result["metrics"].get("force_vec"))]

    # ---------- adaptive phases ----------
    for phase in range(1, N_PHASES + 1):
        print(f"\n=== PHASE {phase} (adaptive) ===")
        # ensure the (array-based) latest result carries the original cylinder geom
        results[-1]["cylinder_geom"] = cylinder_geom
        new_pts = propose_adaptive_points(
            results[-1],
            previous_coords=accumulated,     # clear of ALL prior points
            adaptive_config=ADAPTIVE_CFG,
        )
        if len(new_pts) == 0:
            print("[main] no new points proposed - stopping early.")
            break

        accumulated = np.vstack([accumulated, new_pts])
        print(f"[main] accumulated training points: {len(accumulated)}")

        # re-run keeping every previous point, just with the augmented set
        result = run_gpr(
            **COMMON,
            sample_method="array",
            samples=accumulated,             # method='array' takes explicit coords
            cylinder_geom_override=cylinder_geom,  # shell/face RMSE + momentum force here too
        )
        result["cylinder_geom"] = cylinder_geom   # keep geometry available downstream
        save_all(result, out_dir="plots", phase_label=f"phase{phase}", z_slice_target=25, show=False, true_force=TRUE_FORCE)
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
                cylinder_geom_override=cylinder_geom)
            result["cylinder_geom"] = cylinder_geom
            save_all(result, out_dir="plots", phase_label="phase_topcap",
                     z_slice_target=25, show=False, true_force=TRUE_FORCE)
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

    # per-component force-convergence plot (annotated with drone counts)
    plot_force_convergence(force_curve, true_force=TRUE_FORCE, out_dir="plots")

    # ---------- plot the final phase ----------
    plot_all(results[-1], z_slice_target=25, show=True)
    return results


if __name__ == "__main__":
    main()

"""
sensitivity_study.py
====================

Standalone driver (imports run_gpr / adaptive, modifies nothing) that runs the
adaptive pipeline under several NAMED configurations and compares how fast and
how accurately each one converges the momentum force - focused on Fx/Fy only,
since Fz is dominated by the open-control-volume error and isn't informative.

Why named configs rather than a full grid sweep:
  one phase ~ 30-60 s at res=50, and 5 phases ~ 4-5 min per config. A full grid
  over even 3 params x 3 values would be 27 configs x 5 min ~ 2+ hours. A short
  list of deliberately-chosen configs (baseline + one-param-varied variants)
  gives interpretable "which knob helps" answers in ~30 min. Edit CONFIGS below
  to add/remove. Each config runs phase 0 (80 drones) + N_PHASES adaptive
  phases (80 each), exactly like main.py - phase 0 is NOT bumped to 160.

Outputs:
  - console table: per-config Fx/Fy error vs #drones each phase
  - plots/sensitivity_fx_fy.pdf: Fx and Fy convergence curves, one line per config
  - returns the results dict for further inspection
"""

import os
import gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from GPR import run_gpr
from adaptive import propose_adaptive_points


# ---------------------------------------------------------------------------
# Shared settings - Base configs adapted directly from main.py
# ---------------------------------------------------------------------------
COMMON = dict(
    stl_filepath="input_stls/Aerospecial_building4.stl",
    cfd_filepath="inputs/csv_with_everything.pkl",
    stl_scale=1.0 / 1000.0,
    res=50,
    v_inf=(0.0, 13.6, 0.0),
    bounds_input=np.array([[-100, 100], [30, 275], [0, 80]]),
    n_restarts=6,
    fit_pressure=True,
    posterior_batch=100,
    compute_variance=True,
    var_res=50,
)

TRUE_FORCE = np.array([155433.0, 208647.0, 72586.0])

N_PHASES = 4            # up to 5 adaptive phases to watch for convergence
DRONES_PER_PHASE = 80   # phase 0 and every adaptive phase add this many

# Base adaptive config (matches main.py defaults). Each named config below is
# this dict with a few keys overridden, so differences are attributable.
BASE_ADAPTIVE = dict(
    w_var=0.2, w_grad=0.4, w_vort=0.4,
    pool_size=4000, resample_size=600, score_beta=2.0,
    shell_thick_in=0.20, shell_thick_out=0.20,
    front_frac=0.5, front_half_angle_deg=60.0,
    n_new=DRONES_PER_PHASE,
    frac_region1=0.70, frac_region2=0.15, frac_region3=0.15,
    excl_horizontal=4.2, excl_vertical=4.2,
    spread_radius=None,
)

# Initial cylinder sampling. r_factor/h_factor live here (sample_config), so
# studying them means varying INITIAL_SAMPLING per config (see CONFIGS).
BASE_INITIAL = dict(
    sample_method="cylinder",
    sample_config={"r_factor": 1.2, "h_factor": 1.5, "tilt_deg": 10,
                   "n_points": DRONES_PER_PHASE,
                   "front_frac": 0.5, "front_half_angle_deg": 60.0},
)


def _cfg(**overrides):
    c = dict(BASE_ADAPTIVE); c.update(overrides); return c


def _init(**sample_overrides):
    init = {k: (dict(v) if isinstance(v, dict) else v)
            for k, v in BASE_INITIAL.items()}
    init["sample_config"].update(sample_overrides)
    return init


# ---------------------------------------------------------------------------
# The configs to compare. (name, initial_sampling, adaptive_config)
# Edit freely - add/remove rows. Keep the list short; each row is ~4-5 min.
# ---------------------------------------------------------------------------
CONFIGS = [
    ("baseline",            _init(),                    _cfg()),
    # ("r_factor_1.1",        _init(r_factor=1.0),        _cfg()),
    # ("r_factor_1.5",        _init(r_factor=1.4),        _cfg()),
    # ("r_factor_2.0",        _init(r_factor=1.6),        _cfg()),
    # ("shell_asym_in",       _init(),                    _cfg(shell_thick_in=0.35, shell_thick_out=0.10)),
    # ("shell_asym_out",      _init(),                    _cfg(shell_thick_in=0.10, shell_thick_out=0.35)),
    # ("regions_even",        _init(),                    _cfg(frac_region1=0.5, frac_region2=0.25, frac_region3=0.25)),
    # ("more_on_face_reg",    _init(),                    _cfg(frac_region1=0.8, frac_region2=0.1, frac_region3=0.1)),    
    # ("h_factor_1.2",        _init(h_factor=1.2),        _cfg()),   
    # ("neutral_biased",      _init(front_frac=0.25, front_half_angle_deg=45.0), _cfg(front_frac=None)),
    # ("half_angle=45.0",     _init(front_half_angle_deg=45.0), _cfg(front_half_angle_deg=45.0)),
    # ("front_frac=0.8",      _init(),                    _cfg(front_frac=0.8)),    
    # ("spread_6m",           _init(),                    _cfg(spread_radius=6.0)),

    # ("balanced",              _init(),                    _cfg(w_var=0.33, w_grad=0.34, w_vort=0.33)),
    # ("more w_var",            _init(),                    _cfg(w_var=0.6, w_grad=0.2, w_vort=0.2)),
    # ("more w_grad",           _init(),                    _cfg(w_var=0.2, w_grad=0.6, w_vort=0.2)),
    # ("more w_vort",           _init(),                    _cfg(w_var=0.2, w_grad=0.2, w_vort=0.6)),
    # ("less w_var",            _init(),                    _cfg(w_var=0.2, w_grad=0.4, w_vort=0.4)),      
    # ("less w_grad",           _init(),                    _cfg(w_var=0.4, w_grad=0.2, w_vort=0.4)),
    # ("less w_vort",           _init(),                    _cfg(w_var=0.4, w_grad=0.4, w_vort=0.2)),

    ("bigger shell",        _init(),                    _cfg(shell_thick_in=0.4, shell_thick_out=0.4)),
    ("big shell",           _init(),                    _cfg(shell_thick_in=0.3, shell_thick_out=0.3)),    
    ("small shell",         _init(),                    _cfg(shell_thick_in=0.15, shell_thick_out=0.15)),
    ("smaller shell",       _init(),                    _cfg(shell_thick_in=0.075, shell_thick_out=0.075))     

]


def _free_heavy(res):
    """Drop the big arrays + CFD sampler closure from a finished result so the
    21M-point interpolator it pins gets reclaimed before the next run_gpr builds
    a fresh one. Without this, ~35 run_gpr calls across a study accumulate enough
    interpolators to exhaust memory."""
    if res is None:
        return
    for k in ("sample_dat_shi", "test_points", "GPR_posterior", "GPR_variances",
              "means_tests", "cfd_test_vels", "pressure_posterior", "momentum"):
        res.pop(k, None)


def run_one(name, initial_sampling, adaptive_cfg):
    """Run phase 0 + N_PHASES adaptive phases for one config.
    Returns list of (n_drones, force_vec) including phase 0."""
    print(f"\n########## CONFIG: {name} ##########")
    result = run_gpr(**COMMON, **initial_sampling)
    cyl = result.get("cylinder_geom")
    if cyl is None:
        raise RuntimeError(f"[{name}] phase 0 produced no cylinder_geom.")
    accumulated = np.asarray(result["training_coords"], float)
    curve = [(len(accumulated), result["metrics"].get("force_vec"))]

    for phase in range(1, N_PHASES + 1):
        result["cylinder_geom"] = cyl
        new_pts = propose_adaptive_points(
            result, previous_coords=accumulated,
            adaptive_config=adaptive_cfg, verbose=False)
        if len(new_pts) == 0:
            print(f"[{name}] phase {phase}: no new points; stopping early.")
            break
        accumulated = np.vstack([accumulated, new_pts])
        
        # free the previous phase's heavy data before the next run_gpr builds a
        # fresh 21M-point interpolator (prevents cross-phase memory growth).
        _free_heavy(result)
        gc.collect()
        
        result = run_gpr(**COMMON, sample_method="array",
                         samples=accumulated, cylinder_geom_override=cyl)
        curve.append((len(accumulated), result["metrics"].get("force_vec")))
        fv = result["metrics"].get("force_vec")
        if fv is not None:
            ex = abs(fv[0] - TRUE_FORCE[0]) / abs(TRUE_FORCE[0])
            ey = abs(fv[1] - TRUE_FORCE[1]) / abs(TRUE_FORCE[1])
            print(f"[{name}] phase {phase}  {len(accumulated)} drones  "
                  f"relx={ex:.4g}  rely={ey:.4g}")
                  
    # free the final phase too before moving to the next config
    _free_heavy(result)
    gc.collect()
    return curve


def _fxfy_err(force_vec):
    """In-plane (Fx,Fy) relative error magnitude vs TRUE_FORCE, ignoring Fz."""
    if force_vec is None:
        return np.nan
    fv = np.asarray(force_vec, float)[:2]
    tf = TRUE_FORCE[:2]
    return np.linalg.norm(fv - tf) / np.linalg.norm(tf)


def main():
    all_curves = {}
    for name, init, cfg in CONFIGS:
        try:
            all_curves[name] = run_one(name, init, cfg)
        except Exception as e:
            print(f"[{name}] FAILED: {e!r}")
            all_curves[name] = []

    # ---- console comparison table (Fx/Fy only) ----
    print("\n\n=== Fx/Fy convergence comparison (relative in-plane error) ===")
    print(f"  true Fx={TRUE_FORCE[0]:.6g}  Fy={TRUE_FORCE[1]:.6g}  "
          f"(Fz ignored - open surface)")
    for name, curve in all_curves.items():
        if not curve:
            print(f"  {name:18s}: (failed)")
            continue
        cells = []
        for n_d, fv in curve:
            cells.append(f"{n_d}:{_fxfy_err(fv):.3g}")
        print(f"  {name:18s}: " + "  ".join(cells))

    # ---- convergence plot: Fx and Fy separately, one line per config ----
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    for name, curve in all_curves.items():
        if not curve:
            continue
        ns = [c[0] for c in curve]
        fx = [c[1][0] if c[1] is not None else np.nan for c in curve]
        fy = [c[1][1] if c[1] is not None else np.nan for c in curve]
        axes[0].plot(ns, fx, "-o", label=name, markersize=5)
        axes[1].plot(ns, fy, "-o", label=name, markersize=5)
    axes[0].axhline(TRUE_FORCE[0], ls="--", color="0.5", label="true Fx")
    axes[1].axhline(TRUE_FORCE[1], ls="--", color="0.5", label="true Fy")
    axes[0].set_ylabel("Fx [N]"); axes[1].set_ylabel("Fy [N]")
    axes[1].set_xlabel("number of training drones")
    axes[0].set_title("Sensitivity study: Fx / Fy convergence")
    for ax in axes:
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    out = os.path.join("plots", "sensitivity_fx_fy.pdf")
    fig.savefig(out)
    print(f"\nsaved {out}")
    return all_curves


if __name__ == "__main__":
    main()
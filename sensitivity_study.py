import os
import gc
import numpy as np
import matplotlib.pyplot as plt

from GPR import run_gpr
from adaptive import propose_adaptive_points, propose_top_cap_points


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

N_PHASES = 5            # up to 5 adaptive phases to watch for convergence
DRONES_PER_PHASE = 80   # phase 0 and every adaptive phase add this many

PLOT_DIR = "plots"      # output directory for the PNG figures (e.g. "plots_sens")

# Initial cylinder sampling. r_factor/h_factor live here (sample_config), so
# studying them means varying INITIAL_SAMPLING per config (see CONFIGS).
BASE_INITIAL = dict(
    sample_method="cylinder",
    sample_config={"r_factor": 1.2, "h_factor": 1.5, "tilt_deg": 10,
                   "n_points": DRONES_PER_PHASE, "front_frac": 0.25, "front_half_angle_deg": 45.0},
)

# Base adaptive config (matches main.py defaults). Each named config below is
# this dict with a few keys overridden, so differences are attributable.
BASE_ADAPTIVE = dict(
    # difficulty score (computed on the existing res^3 grid via np.gradient)
    w_var=0.2, w_grad=0.4, w_vort=0.4,   # favor gradient + vorticity
    # weighted-LHS candidate pool over the thick tilted cylinder shell
    pool_size=4000, resample_size=600, score_beta=2.0,
    shell_thick_in=0.20, shell_thick_out=0.20,   # shell spans 0.7R .. 1.3R
    front_frac=0.5, front_half_angle_deg=60.0,   # bias toward the wake side
    # per-phase budget (mostly on-cylinder)
    n_new=DRONES_PER_PHASE, frac_region1=0.50, frac_region2=0.35, frac_region3=0.15,
    # spacing: hard drone limit + optional spread relaxation
    excl_horizontal=1.2, excl_vertical=4.2,
    spread_radius=None,   # set e.g. 6.0 to relax points ~6 m apart laterally
)

# Optional FINAL top-cap phase per config. Off by default since this study is
# Fx/Fy-focused and the cap mainly affects Fz (the open-top flux), but useful
# if you want to see the cap's effect on a curve. Appended as one extra point.
TOP_CAP_PHASE = False
TOP_CAP_N = 30            # number of cap drones
TOP_CAP_Z_OFFSET = 0.0    # place cap at cylinder z_top + this

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
    # ("baseline",            _init(),                    _cfg()),
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

    # ("balanced",             _init(),                    _cfg(w_var=0.33, w_grad=0.34, w_vort=0.33)),
    # ("more w_var",           _init(),                    _cfg(w_var=0.6, w_grad=0.2, w_vort=0.2)),
    # ("more w_grad",          _init(),                    _cfg(w_var=0.2, w_grad=0.6, w_vort=0.2)),
    # ("more w_vort",          _init(),                    _cfg(w_var=0.2, w_grad=0.2, w_vort=0.6)),
    # ("less w_var",           _init(),                    _cfg(w_var=0.2, w_grad=0.4, w_vort=0.4)),      
    # ("less w_grad",          _init(),                    _cfg(w_var=0.4, w_grad=0.2, w_vort=0.4)),
    # ("less w_vort",          _init(),                    _cfg(w_var=0.4, w_grad=0.4, w_vort=0.2)),

    # ("bigger shell",        _init(),                    _cfg(shell_thick_in=0.4, shell_thick_out=0.4)),
    # ("big shell",           _init(),                    _cfg(shell_thick_in=0.3, shell_thick_out=0.3)),    
    # ("small shell",         _init(),                    _cfg(shell_thick_in=0.15, shell_thick_out=0.15)),
    # ("smaller shell",       _init(),                    _cfg(shell_thick_in=0.075, shell_thick_out=0.075))     

    # ("1 drones per step",  _init(n_points=1),         _cfg(n_new=1)),
    # ("5 drones per step",  _init(n_points=5),         _cfg(n_new=5)),    
    # ("10 drones per step",  _init(n_points=10),         _cfg(n_new=10)),
    # ("20 drones per step",  _init(n_points=20),         _cfg(n_new=20)),
    # ("30 drones per step",  _init(n_points=30),         _cfg(n_new=30)),

    # ("40 drones per step",  _init(n_points=40),         _cfg(n_new=40)),
    # ("50 drones per step",  _init(n_points=50),         _cfg(n_new=50)),     
    ("60 drones per step",  _init(n_points=60),         _cfg(n_new=60)),
    ("70 drones per step",  _init(n_points=70),         _cfg(n_new=70)),
    ("80 drones per step",  _init(n_points=80),         _cfg(n_new=80)),
    ("90 drones per step",  _init(n_points=90),         _cfg(n_new=90)),
    ("100 drones per step", _init(n_points=100),        _cfg(n_new=100)),
    ("150 drones per step", _init(n_points=150),        _cfg(n_new=150)),
    ("200 drones per step", _init(n_points=200),        _cfg(n_new=200))
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
            
    if TOP_CAP_PHASE:
        result["cylinder_geom"] = cyl
        cap_pts = propose_top_cap_points(
            result, previous_coords=accumulated,
            n_new=TOP_CAP_N, cap_z_offset=TOP_CAP_Z_OFFSET,
            adaptive_config=adaptive_cfg, verbose=False)
        if len(cap_pts):
            accumulated = np.vstack([accumulated, cap_pts])
            _free_heavy(result)
            gc.collect()
            result = run_gpr(**COMMON, sample_method="array",
                             samples=accumulated, cylinder_geom_override=cyl)
            curve.append((len(accumulated), result["metrics"].get("force_vec")))
            fv = result["metrics"].get("force_vec")
            if fv is not None:
                ex = abs(fv[0] - TRUE_FORCE[0]) / abs(TRUE_FORCE[0])
                ey = abs(fv[1] - TRUE_FORCE[1]) / abs(TRUE_FORCE[1])
                print(f"[{name}] top-cap  {len(accumulated)} drones  "
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


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
BAND = 0.05   # +-5% band drawn on every plot

def _series(curve, comp):
    """Return (ns, values) for force component `comp` (0=Fx, 1=Fy)."""
    ns = [c[0] for c in curve]
    vals = [c[1][comp] if c[1] is not None else np.nan for c in curve]
    return ns, vals

def _relerr_series(curve, comp):
    """Return (ns, signed relative error in %) for component `comp`."""
    ns = [c[0] for c in curve]
    tf = TRUE_FORCE[comp]
    vals = [100.0 * (c[1][comp] - tf) / abs(tf) if c[1] is not None else np.nan
            for c in curve]
    return ns, vals


def _plot_abs_component(all_curves, comp, label, fname):
    """Single-component absolute force convergence with +-5% band."""
    true = TRUE_FORCE[comp]
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, curve in all_curves.items():
        if not curve:
            continue
        ns, vals = _series(curve, comp)
        ax.plot(ns, vals, "-o", label=name, markersize=5)
    ax.axhline(true, ls="--", color="0.4", label=f"true {label}")
    ax.axhspan(true * (1 - BAND), true * (1 + BAND), color="0.5", alpha=0.15,
               label="+-5% band")
    ax.set_xlabel("number of training drones")
    ax.set_ylabel(f"{label} [N]")
    ax.set_title(f"Sensitivity study: {label} convergence (+-5% band)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    return fig


def _plot_abs_combined(all_curves, fname):
    """Fx and Fy absolute force convergence stacked, each with +-5% band."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    for comp, (ax, label) in enumerate(zip(axes, ("Fx", "Fy"))):
        true = TRUE_FORCE[comp]
        for name, curve in all_curves.items():
            if not curve:
                continue
            ns, vals = _series(curve, comp)
            ax.plot(ns, vals, "-o", label=name, markersize=5)
        ax.axhline(true, ls="--", color="0.4", label=f"true {label}")
        ax.axhspan(true * (1 - BAND), true * (1 + BAND), color="0.5",
                   alpha=0.15, label="+-5% band")
        ax.set_ylabel(f"{label} [N]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[1].set_xlabel("number of training drones")
    axes[0].set_title("Sensitivity study: Fx / Fy convergence (+-5% band)")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    return fig


def _plot_relerr_component(all_curves, comp, label, fname):
    """Single-component signed relative error (%) with +-5% band."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, curve in all_curves.items():
        if not curve:
            continue
        ns, vals = _relerr_series(curve, comp)
        ax.plot(ns, vals, "-o", label=name, markersize=5)
    ax.axhline(0.0, ls="--", color="0.4", label="exact")
    ax.axhspan(-100 * BAND, 100 * BAND, color="0.5", alpha=0.15,
               label="+-5% band")
    ax.set_xlabel("number of training drones")
    ax.set_ylabel(f"{label} relative error [%]")
    ax.set_title(f"Sensitivity study: {label} relative error (+-5% band)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    return fig

def _plot_relerr_per_phase(all_curves, fname):
    """In-plane (Fx,Fy) relative error (%) vs phase index, one line per config,
    with +-5% band. Phase 0 is the initial run; 1..N are adaptive phases."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, curve in all_curves.items():
        if not curve:
            continue
        phases = list(range(len(curve)))
        vals = [100.0 * _fxfy_err(fv) for _, fv in curve]
        ax.plot(phases, vals, "-o", label=name, markersize=5)
    ax.axhline(0.0, ls="--", color="0.4", label="exact")
    ax.axhspan(0.0, 100 * BAND, color="0.5", alpha=0.15, label="+-5% band")
    ax.set_xlabel("phase index (0 = initial, 1..N = adaptive)")
    ax.set_ylabel("in-plane (Fx,Fy) relative error [%]")
    ax.set_title("Sensitivity study: Fx/Fy relative error per phase (+-5% band)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    return fig

def _plot_relerr_combined(all_curves, fname):
    """Fx and Fy signed relative error (%) stacked, each with +-5% band."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    for comp, (ax, label) in enumerate(zip(axes, ("Fx", "Fy"))):
        for name, curve in all_curves.items():
            if not curve:
                continue
            ns, vals = _relerr_series(curve, comp)
            ax.plot(ns, vals, "-o", label=name, markersize=5)
        ax.axhline(0.0, ls="--", color="0.4", label="exact")
        ax.axhspan(-100 * BAND, 100 * BAND, color="0.5", alpha=0.15,
                   label="+-5% band")
        ax.set_ylabel(f"{label} rel. error [%]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[1].set_xlabel("number of training drones")
    axes[0].set_title("Sensitivity study: Fx / Fy relative error (+-5% band)")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    return fig


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

    # ---- generate all plots as PNG ----
    os.makedirs(PLOT_DIR, exist_ok=True)
    figs = []

    figs.append(_plot_abs_combined(
        all_curves, os.path.join(PLOT_DIR, "sensitivity_fx_fy.png")))
    figs.append(_plot_abs_component(
        all_curves, 0, "Fx", os.path.join(PLOT_DIR, "sensitivity_fx.png")))
    figs.append(_plot_abs_component(
        all_curves, 1, "Fy", os.path.join(PLOT_DIR, "sensitivity_fy.png")))

    figs.append(_plot_relerr_combined(
        all_curves, os.path.join(PLOT_DIR, "sensitivity_fx_fy_relerr.png")))
    figs.append(_plot_relerr_component(
        all_curves, 0, "Fx", os.path.join(PLOT_DIR, "sensitivity_fx_relerr.png")))
    figs.append(_plot_relerr_component(
        all_curves, 1, "Fy", os.path.join(PLOT_DIR, "sensitivity_fy_relerr.png")))

    figs.append(_plot_relerr_per_phase(
        all_curves, os.path.join(PLOT_DIR, "sensitivity_relerr_per_phase.png")))
    
    for f in ("sensitivity_fx_fy", "sensitivity_fx", "sensitivity_fy",
              "sensitivity_fx_fy_relerr", "sensitivity_fx_relerr",
              "sensitivity_fy_relerr"):
        print(f"saved {os.path.join(PLOT_DIR, f + '.png')}")

    # ---- show them all on screen at the end of the run ----
    plt.show()

    return all_curves


if __name__ == "__main__":
    main()
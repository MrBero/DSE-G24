import os
import gc
import numpy as np
import matplotlib.pyplot as plt

from GPR import run_gpr
from adaptive import propose_adaptive_points


# ---------------------------------------------------------------------------
# Shared settings - identical to the main sensitivity study config
# ---------------------------------------------------------------------------
COMMON = dict(
    stl_filepath="input_stls/Aerospecial_building4.stl",
    cfd_filepath="inputs/csv_with_everything.pkl",
    stl_scale=1.0 / 1000.0,
    res=30,
    v_inf=(0.0, 13.6, 0.0),
    bounds_input=np.array([[-100, 100], [30, 275], [0, 80]]),
    n_restarts=6,
    fit_pressure=True,
    posterior_batch=100,
    compute_variance=True,
    var_res=50,
)

TRUE_FORCE = np.array([155433.0, 208647.0, 72586.0])

PLOT_DIR = "plots_budget"
SEEDS = [7, 42, 67, 69, 3, 420, 1234, 13]
BAND = 0.05                          # +-5% acceptance band drawn on plots

# Budget allocations: (label, n_per_phase, n_phases). total ~= n_per_phase*(n_phases+1)
# including phase 0. We hold the PER-PHASE budget constant across phase 0 and the
# adaptive phases (phase 0 = initial cylinder of n_per_phase, then n_phases
# adaptive batches of n_per_phase), so total nominal = n_per_phase*(n_phases+1).
ALLOCATIONS = [
    ("67x6",  67,  6),
    ("80x5",  80,  5),
    ("100x4", 100, 4),
    ("133x3", 133, 3),
    ("200x2", 200, 2),
]

# Base adaptive config (matches the main sensitivity study). n_new is overridden
# per allocation; seed/pool_seed set per run.
BASE_ADAPTIVE = dict(
    # difficulty score (computed on the existing res^3 grid via np.gradient)
    w_var=0.2, w_grad=0.4, w_vort=0.4,   # favor gradient + vorticity
    # weighted-LHS candidate pool over the thick tilted cylinder shell
    pool_size=4000, resample_size=600, score_beta=2.0,
    shell_thick_in=0.20, shell_thick_out=0.20,   # shell spans 0.7R .. 1.3R
    front_frac=0.5, front_half_angle_deg=60.0,   # bias toward the wake side
    # per-phase budget (mostly on-cylinder)
    n_new=100, frac_region1=0.70, frac_region2=0.15, frac_region3=0.15,
    # spacing: hard drone limit + optional spread relaxation
    excl_horizontal=1.2, excl_vertical=4.2,
    spread_radius=None,   # set e.g. 6.0 to relax points ~6 m apart laterally
)

# Base initial cylinder (matches the main study). n_points/top_cap_frac set per
# allocation; seed set per run.
BASE_INITIAL = dict(
    sample_method="cylinder",
    sample_config={"r_factor": 1.2, "h_factor": 1.5, "tilt_deg": 10,
                   "front_frac": 0.25, "front_half_angle_deg": 45,
                   "top_cap": True},
)


def _topcap_frac(n_per_phase):
    """Initial-cylinder top-cap fraction: max(20/n, 0.20) -> >=20 drones or 20%,
    whichever is larger, carved OUT of the phase-0 budget."""
    return max(20.0 / n_per_phase, 0.20)
 
 
def _init(n_per_phase, seed):
    cfg = {k: (dict(v) if isinstance(v, dict) else v)
           for k, v in BASE_INITIAL.items()}
    cfg["sample_config"].update(
        n_points=n_per_phase,
        top_cap_frac=_topcap_frac(n_per_phase),
        seed=seed,
    )
    return cfg
 
 
def _cfg(n_per_phase, seed):
    c = dict(BASE_ADAPTIVE)
    c.update(n_new=n_per_phase, seed=seed)
    return c
 
 
def _free_heavy(res):
    if res is None:
        return
    for k in ("sample_dat_shi", "test_points", "GPR_posterior", "GPR_variances",
              "means_tests", "cfd_test_vels", "pressure_posterior", "momentum"):
        res.pop(k, None)
 
 
def run_one(label, n_per_phase, n_phases, seed):
    """Phase 0 (initial cylinder, n_per_phase incl. top cap) + n_phases adaptive
    phases of n_per_phase each. Returns list of (n_accumulated, force_vec)."""
    print(f"\n########## {label}  seed={seed} ##########")
    initial = _init(n_per_phase, seed)
    adaptive_cfg = _cfg(n_per_phase, seed)
 
    result = run_gpr(**COMMON, **initial)
    cyl = result.get("cylinder_geom")
    if cyl is None:
        raise RuntimeError(f"[{label}/{seed}] phase 0 produced no cylinder_geom.")
    accumulated = np.asarray(result["training_coords"], float)
    curve = [(len(accumulated), result["metrics"].get("force_vec"))]
    fv = result["metrics"].get("force_vec")
    if fv is not None:
        ex = abs(fv[0] - TRUE_FORCE[0]) / abs(TRUE_FORCE[0])
        ey = abs(fv[1] - TRUE_FORCE[1]) / abs(TRUE_FORCE[1])
        print(f"[{label}/{seed}] phase 0  {len(accumulated)} pts  "
              f"relx={ex:.4g}  rely={ey:.4g}")
    for phase in range(1, n_phases):
        result["cylinder_geom"] = cyl
        # fresh candidate lattice per phase (see module docstring)
        phase_cfg = dict(adaptive_cfg)
        phase_cfg["pool_seed"] = seed + phase
        new_pts = propose_adaptive_points(
            result, previous_coords=accumulated,
            adaptive_config=phase_cfg, verbose=False)
        if len(new_pts) == 0:
            print(f"[{label}/{seed}] phase {phase}: no new points; stopping early.")
            break
        accumulated = np.vstack([accumulated, new_pts])
 
        _free_heavy(result)
        gc.collect()
 
        result = run_gpr(**COMMON, sample_method="array",
                         samples=accumulated, cylinder_geom_override=cyl)
        curve.append((len(accumulated), result["metrics"].get("force_vec")))
        fv = result["metrics"].get("force_vec")
        if fv is not None:
            ex = abs(fv[0] - TRUE_FORCE[0]) / abs(TRUE_FORCE[0])
            ey = abs(fv[1] - TRUE_FORCE[1]) / abs(TRUE_FORCE[1])
            print(f"[{label}/{seed}] phase {phase}  {len(accumulated)} pts  "
                  f"relx={ex:.4g}  rely={ey:.4g}")
 
    _free_heavy(result)
    gc.collect()
    return curve
 
 
# ---------------------------------------------------------------------------
# Aggregation: average PER PHASE STEP across seeds (exact - every seed of a
# config has the same phases). The averaged point is plotted at the MEAN
# accumulated drone count for that phase step. No interpolation needed.
# ---------------------------------------------------------------------------
def _component(fv, comp):
    return np.nan if fv is None else float(np.asarray(fv, float)[comp])
 
 
def _relerr(fv, comp):
    if fv is None:
        return np.nan
    return 100.0 * (np.asarray(fv, float)[comp] - TRUE_FORCE[comp]) / abs(TRUE_FORCE[comp])
 
 
def _inplane_relerr(fv):
    if fv is None:
        return np.nan
    d = np.asarray(fv, float)[:2] - TRUE_FORCE[:2]
    return 100.0 * np.linalg.norm(d) / np.linalg.norm(TRUE_FORCE[:2])
 
 
def _aggregate(all_runs, n_per_phase, n_phases, fn):
    """Average fn(force_vec) across seeds at each PHASE STEP (phase 0..n_phases).
    Returns (x, mean, std, x_std) where:
       x      = mean accumulated drone count across seeds at that phase
       mean   = mean of fn across seeds at that phase
       std    = std of fn across seeds (the +-1 std band, y-direction)
       x_std  = std of accumulated count across seeds (optional x error bar)
    Each seed's curve has len == n_phases+1 (phase 0 + adaptive phases), so phase
    step k is simply curve[k] - exact alignment, no interpolation. A seed that
    stopped early contributes nan for the missing tail phases and is dropped from
    those steps by nanmean."""
    runs = [c for c in all_runs if c]
    n_steps = n_phases
    counts = np.full((len(runs), n_steps), np.nan)   # accumulated drone count
    vals = np.full((len(runs), n_steps), np.nan)     # fn value
    for i, c in enumerate(runs):
        for k in range(min(len(c), n_steps)):
            counts[i, k] = c[k][0]
            vals[i, k] = fn(c[k][1])
    with np.errstate(invalid="ignore"):
        x = np.nanmean(counts, axis=0)
        x_std = np.nanstd(counts, axis=0)
        mean = np.nanmean(vals, axis=0)
        std = np.nanstd(vals, axis=0)
    return x, mean, std, x_std
 
 
def _print_full_summary(results_by_cfg):
    """Per-config, per-phase numeric dump: mean accumulated count, and mean +-std
    of signed relative error [%] for Fx and Fy across seeds. Printed so all the
    info is available even if the figures are hard to read."""
    print("\n\n=== full numeric summary: rel. error [%], mean +- std over seeds, "
          "per phase ===")
    for label, runs, npp, nph in results_by_cfg:
        x, mx, sx, xstd = _aggregate(runs, npp, nph, lambda fv: _relerr(fv, 0))
        _, my, sy, _ = _aggregate(runs, npp, nph, lambda fv: _relerr(fv, 1))
        _, mip, sip, _ = _aggregate(runs, npp, nph, _inplane_relerr)
        n_seeds = sum(1 for c in runs if c)
        print(f"\n  --- {label}  ({npp}/phase, {nph} phases incl. phase 0, "
              f"{n_seeds} seeds) ---")
        print(f"    {'phase':>5}  {'~pts':>7} {'+-':>5}   "
              f"{'Fx%':>8} {'+-':>6}   {'Fy%':>8} {'+-':>6}   "
              f"{'inplane%':>9} {'+-':>6}")
        for k in range(len(x)):
            print(f"    {k:>5}  {x[k]:7.1f} {xstd[k]:5.1f}   "
                  f"{mx[k]:+8.3f} {sx[k]:6.3f}   "
                  f"{my[k]:+8.3f} {sy[k]:6.3f}   "
                  f"{mip[k]:9.3f} {sip[k]:6.3f}")
 
 
# ---------------------------------------------------------------------------
# Plotting: one band (mean +- 1 std) per config
# ---------------------------------------------------------------------------
def _band_plot(ax, results_by_cfg, fn, ylabel, title, hline=None, band_abs=None,
               band_rel_true=None, x_err=False):
    """Draw a mean line + +-1std fill per config. fn -> scalar from force_vec.
    Points sit at the mean accumulated count per phase step; set x_err=True to
    also draw the across-seed spread in count as horizontal error bars."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_by_cfg)))
    for (label, runs, npp, nph), color in zip(results_by_cfg, colors):
        x, mean, std, x_std = _aggregate(runs, npp, nph, fn)
        ax.plot(x, mean, "-o", color=color, label=label, markersize=4)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
        if x_err:
            ax.errorbar(x, mean, xerr=x_std, fmt="none", ecolor=color,
                        alpha=0.5, capsize=2)
    if hline is not None:
        ax.axhline(hline, ls="--", color="0.4")
    if band_abs is not None:           # absolute force: +-5% of true around true
        ax.axhspan(band_abs * (1 - BAND), band_abs * (1 + BAND),
                   color="0.5", alpha=0.12, label="+-5% band")
    if band_rel_true is not None:      # rel-error plot: +-5% band around 0
        ax.axhspan(-100 * BAND, 100 * BAND, color="0.5", alpha=0.12,
                   label="+-5% band")
    ax.set_xlabel("accumulated sampling points (mean per phase)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
 
 
def _save(fig, fname):
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"saved {fname}")
 
 
def _make_band_figs(results_by_cfg):
    figs = []
 
    # absolute Fx with band
    fig, ax = plt.subplots(figsize=(9, 5))
    _band_plot(ax, results_by_cfg, lambda fv: _component(fv, 0),
               "Fx [N]", "Budget study: Fx convergence (mean +-1 std)",
               hline=TRUE_FORCE[0], band_abs=TRUE_FORCE[0])
    _save(fig, os.path.join(PLOT_DIR, "budget_fx_band.png")); figs.append(fig)
 
    # absolute Fy with band
    fig, ax = plt.subplots(figsize=(9, 5))
    _band_plot(ax, results_by_cfg, lambda fv: _component(fv, 1),
               "Fy [N]", "Budget study: Fy convergence (mean +-1 std)",
               hline=TRUE_FORCE[1], band_abs=TRUE_FORCE[1])
    _save(fig, os.path.join(PLOT_DIR, "budget_fy_band.png")); figs.append(fig)
 
    # Fx & Fy stacked
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    _band_plot(axes[0], results_by_cfg, lambda fv: _component(fv, 0),
               "Fx [N]", "Budget study: Fx / Fy convergence (mean +-1 std)",
               hline=TRUE_FORCE[0], band_abs=TRUE_FORCE[0])
    _band_plot(axes[1], results_by_cfg, lambda fv: _component(fv, 1),
               "Fy [N]", "", hline=TRUE_FORCE[1], band_abs=TRUE_FORCE[1])
    _save(fig, os.path.join(PLOT_DIR, "budget_fx_fy_band.png")); figs.append(fig)
 
    # signed rel error Fx
    fig, ax = plt.subplots(figsize=(9, 5))
    _band_plot(ax, results_by_cfg, lambda fv: _relerr(fv, 0),
               "Fx relative error [%]",
               "Budget study: Fx relative error (mean +-1 std)",
               hline=0.0, band_rel_true=True)
    _save(fig, os.path.join(PLOT_DIR, "budget_fx_relerr_band.png")); figs.append(fig)
 
    # signed rel error Fy
    fig, ax = plt.subplots(figsize=(9, 5))
    _band_plot(ax, results_by_cfg, lambda fv: _relerr(fv, 1),
               "Fy relative error [%]",
               "Budget study: Fy relative error (mean +-1 std)",
               hline=0.0, band_rel_true=True)
    _save(fig, os.path.join(PLOT_DIR, "budget_fy_relerr_band.png")); figs.append(fig)
 
    # in-plane |Fx,Fy| rel error
    fig, ax = plt.subplots(figsize=(9, 5))
    _band_plot(ax, results_by_cfg, _inplane_relerr,
               "in-plane (Fx,Fy) rel error [%]",
               "Budget study: in-plane relative error (mean +-1 std)",
               hline=0.0)
    ax.axhspan(0.0, 100 * BAND, color="0.5", alpha=0.12, label="+-5% band")
    _save(fig, os.path.join(PLOT_DIR, "budget_relerr_inplane_band.png")); figs.append(fig)
 
    return figs
 
 
def _make_per_config_figs(results_by_cfg):
    """One detailed figure per config: Fx and Fy abs convergence with band,
    plus every individual seed line faint underneath."""
    sub = os.path.join(PLOT_DIR, "per_config")
    os.makedirs(sub, exist_ok=True)
    figs = []
    for label, runs, npp, nph in results_by_cfg:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        for comp, (ax, name) in enumerate(zip(axes, ("Fx", "Fy"))):
            # faint individual seeds
            for c in runs:
                if not c:
                    continue
                ns = [p[0] for p in c]
                vs = [_component(p[1], comp) for p in c]
                ax.plot(ns, vs, "-", color="0.7", lw=0.8, alpha=0.6)
            # band (mean +-1 std across seeds, per phase step)
            x, mean, std, x_std = _aggregate(runs, npp, nph,
                                             lambda fv, cc=comp: _component(fv, cc))
            ax.plot(x, mean, "-o", color="tab:blue", markersize=4,
                    label="mean over seeds")
            ax.fill_between(x, mean - std, mean + std,
                            color="tab:blue", alpha=0.2, label="+-1 std")
            true = TRUE_FORCE[comp]
            ax.axhline(true, ls="--", color="0.4", label=f"true {name}")
            ax.axhspan(true * (1 - BAND), true * (1 + BAND),
                       color="0.5", alpha=0.12, label="+-5% band")
            ax.set_ylabel(f"{name} [N]")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="best")
        axes[1].set_xlabel("accumulated sampling points (mean per phase)")
        axes[0].set_title(f"Budget config {label}: Fx / Fy ({len(SEEDS)} seeds, mean +-1 std)")
        fname = os.path.join(sub, f"{label}.png")
        _save(fig, fname); figs.append(fig)
    return figs
 
 
def main():
    # run every (allocation x seed)
    results_by_cfg = []   # list of (label, [curve_per_seed], n_per_phase, n_phases)
    for label, npp, nph in ALLOCATIONS:
        runs = []
        for seed in SEEDS:
            try:
                runs.append(run_one(label, npp, nph, seed))
            except Exception as e:
                print(f"[{label}/{seed}] FAILED: {e!r}")
                runs.append([])
        results_by_cfg.append((label, runs, npp, nph))
 
    # ---- console table: per-config per-seed in-plane error at final phase ----
    print("\n\n=== final-phase in-plane (Fx,Fy) relative error [%], per seed ===")
    for label, runs, npp, nph in results_by_cfg:
        cells = []
        for seed, c in zip(SEEDS, runs):
            if not c:
                cells.append(f"s{seed}:fail")
                continue
            cells.append(f"s{seed}:{_inplane_relerr(c[-1][1]):.2f}")
        print(f"  {label:8s}: " + "  ".join(cells))
 
    # ---- full numeric summary (mean +- std per phase, all configs) ----
    _print_full_summary(results_by_cfg)
 
    # ---- plots ----
    os.makedirs(PLOT_DIR, exist_ok=True)
    _make_band_figs(results_by_cfg)
    _make_per_config_figs(results_by_cfg)
 
    plt.show()
    return results_by_cfg
 
 
if __name__ == "__main__":
    main()
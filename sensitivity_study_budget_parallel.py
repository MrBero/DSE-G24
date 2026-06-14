"""
sensitivity_study_budget_parallel.py
====================================

Parallel version of sensitivity_study_budget.py.

Prerequisite - WARM CACHE, SO RUN 1 MAIN WITH THE SAME BASE CONFIG 1ST!
  Each run_gpr reads prior_cache/*.pkl (grid + momentum priors, keyed by
  res/v_inf/geometry, NOT by seed). On a cold cache multiple workers would race
  to create the same file and corrupt it. RUN ONCE WITH 1 PROCESS FIRST so the
  caches exist; after that every worker only READS them (concurrent reads are
  safe) and skips the heavy Julia grid-prior sweep. 
  
"""

import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    grid_eval=False,
)

TRUE_FORCE = np.array([155433.0, 208647.0, 72586.0])

PLOT_DIR = "plots_budget"
SEEDS = [7, 42, 67, 420, 1234, 15, 4321, 1324, 4213, 3, 696, 6767]         # 5 seeds per config -> 25 runs total
BAND = 0.05                          # +-5% acceptance band drawn on plots
MAX_WORKERS = 6                      # parallel processes; start 2-3, raise cautiously

# Budget allocations: (label, n_per_phase, n_phases). total ~= n_per_phase*n_phases
# (phase 0 included). phase 0 = initial cylinder of n_per_phase, then
# (n_phases-1) adaptive batches of n_per_phase.
ALLOCATIONS = [
    # ("67x6",  67,  6),
    # ("80x5",  80,  5),
    ("100x4", 100, 4),
    # ("133x3", 133, 3),
    # ("160x3", 160, 3),    
    # ("200x2", 200, 2),
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
    fd_step=[1.0,1.0,1.0]
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
    closer = res.get("_close_solver")
    if closer is not None:
        closer()
    for k in ("sample_dat_shi", "test_points", "GPR_posterior", "GPR_variances",
              "means_tests", "cfd_test_vels", "pressure_posterior", "momentum",
              "_prior_fn", "_close_solver", "_chol_c", "_chol_low"):
        res.pop(k, None)
 
 
def run_one(label, n_per_phase, n_phases, seed):
    """Phase 0 (initial cylinder, n_per_phase incl. top cap) + (n_phases-1)
    adaptive phases of n_per_phase each. Returns list of (n_accumulated,
    force_vec). Phases run sequentially - the adaptive part is NOT parallel."""
    print(f"\n########## {label}  seed={seed} ##########", flush=True)
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
              f"relx={ex:.4g}  rely={ey:.4g}", flush=True)
    for phase in range(1, n_phases):
        result["cylinder_geom"] = cyl
        # fresh candidate lattice per phase (see module docstring)
        phase_cfg = dict(adaptive_cfg)
        phase_cfg["pool_seed"] = seed + phase
        new_pts = propose_adaptive_points(
            result, previous_coords=accumulated,
            adaptive_config=phase_cfg, verbose=False)
        if len(new_pts) == 0:
            print(f"[{label}/{seed}] phase {phase}: no new points; stopping early.",
                  flush=True)
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
                  f"relx={ex:.4g}  rely={ey:.4g}", flush=True)
 
    _free_heavy(result)
    gc.collect()
    return curve
 
 
def _job(args):
    """Top-level worker for the process pool: runs one (config, seed) job.
    Must be module-level (picklable). Returns (label, seed, curve)."""
    label, npp, nph, seed = args
    try:
        return (label, seed, run_one(label, npp, nph, seed))
    except Exception as e:
        print(f"[{label}/{seed}] FAILED: {e!r}", flush=True)
        return (label, seed, [])
 
 
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
    """Average fn(force_vec) across seeds at each PHASE STEP (phase 0..n_phases-1).
    Returns (x, mean, std, x_std) where:
       x      = mean accumulated drone count across seeds at that phase
       mean   = mean of fn across seeds at that phase
       std    = std of fn across seeds (the +-1 std band, y-direction)
       x_std  = std of accumulated count across seeds (optional x error bar)
    Each seed's curve has len == n_phases (phase 0 + adaptive phases), so phase
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
               band_rel_true=None, x_err=False, ymax_cap=None, fill_style="dashed"):
    """Draw a mean line +-1std per config. fn -> scalar from force_vec.
    fill_style:
       "dashed" -> +-1std drawn as thin low-opacity dashed boundary lines (no
                   fill). Used for the multi-config figures where overlapping
                   filled bands turn into mud.
       "fill"   -> translucent filled band (fine when one config per axis).
    Points sit at the mean accumulated count per phase step; set x_err=True to
    also draw the across-seed spread in count as horizontal error bars.
    ymax_cap (if set) clamps the y-axis to +-ymax_cap, tightened to the data
    range when everything fits inside a smaller window."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_by_cfg)))
    finite = []
    for (label, runs, npp, nph), color in zip(results_by_cfg, colors):
        x, mean, std, x_std = _aggregate(runs, npp, nph, fn)
        ax.plot(x, mean, "-o", color=color, label=label, markersize=4)
        if fill_style == "fill":
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
        else:   # "dashed": thin low-opacity boundary lines, no fill
            ax.plot(x, mean - std, ls="--", color=color, lw=0.9, alpha=0.4)
            ax.plot(x, mean + std, ls="--", color=color, lw=0.9, alpha=0.4)
        finite += [v for v in (mean - std) if np.isfinite(v)]
        finite += [v for v in (mean + std) if np.isfinite(v)]
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
    if ymax_cap is not None:           # clamp to +-ymax_cap, tighten if data fits
        if finite:
            lim = min(ymax_cap, max(abs(min(finite)), abs(max(finite)),
                                    100 * BAND) * 1.1)
        else:
            lim = ymax_cap
        ax.set_ylim(-lim, lim)
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
               hline=0.0, band_rel_true=True, ymax_cap=50.0)
    _save(fig, os.path.join(PLOT_DIR, "budget_fx_relerr_band.png")); figs.append(fig)
 
    # signed rel error Fy
    fig, ax = plt.subplots(figsize=(9, 5))
    _band_plot(ax, results_by_cfg, lambda fv: _relerr(fv, 1),
               "Fy relative error [%]",
               "Budget study: Fy relative error (mean +-1 std)",
               hline=0.0, band_rel_true=True, ymax_cap=50.0)
    _save(fig, os.path.join(PLOT_DIR, "budget_fy_relerr_band.png")); figs.append(fig)
 
    # in-plane |Fx,Fy| rel error (non-negative magnitude -> one-sided y-cap)
    fig, ax = plt.subplots(figsize=(9, 5))
    _band_plot(ax, results_by_cfg, _inplane_relerr,
               "in-plane (Fx,Fy) rel error [%]",
               "Budget study: in-plane relative error (mean +-1 std)",
               hline=0.0)
    ax.axhspan(0.0, 100 * BAND, color="0.5", alpha=0.12, label="+-5% band")
    # clamp top to 50%, tighten to data when it fits
    top = ax.get_ylim()[1]
    ax.set_ylim(0.0, min(50.0, max(top, 100 * BAND * 1.1)))
    _save(fig, os.path.join(PLOT_DIR, "budget_relerr_inplane_band.png")); figs.append(fig)
 
    return figs
 
 
def _make_per_config_figs(results_by_cfg, ymax_cap=50.0):
    """One detailed figure per config: Fx and Fy RELATIVE-ERROR convergence with
    band, plus every individual seed line faint underneath. The y-axis is capped
    at +-ymax_cap %, but tightened to the data range when everything fits inside
    a smaller window (so the cap is a ceiling, not a forced window)."""
    sub = os.path.join(PLOT_DIR, "per_config")
    os.makedirs(sub, exist_ok=True)
    figs = []
    for label, runs, npp, nph in results_by_cfg:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        for comp, (ax, name) in enumerate(zip(axes, ("Fx", "Fy"))):
            finite = []     # collect all plotted values to size the y-axis
            # faint individual seeds (signed relative error %)
            for c in runs:
                if not c:
                    continue
                ns = [p[0] for p in c]
                vs = [_relerr(p[1], comp) for p in c]
                ax.plot(ns, vs, "-", color="0.7", lw=0.8, alpha=0.6)
                finite += [v for v in vs if np.isfinite(v)]
            # band (mean +-1 std across seeds, per phase step)
            x, mean, std, x_std = _aggregate(runs, npp, nph,
                                             lambda fv, cc=comp: _relerr(fv, cc))
            ax.plot(x, mean, "-o", color="tab:blue", markersize=4,
                    label="mean over seeds")
            ax.fill_between(x, mean - std, mean + std,
                            color="tab:blue", alpha=0.2, label="+-1 std")
            finite += [v for v in (mean - std) if np.isfinite(v)]
            finite += [v for v in (mean + std) if np.isfinite(v)]
            ax.axhline(0.0, ls="--", color="0.4", label="exact")
            ax.axhspan(-100 * BAND, 100 * BAND, color="0.5", alpha=0.12,
                       label="+-5% band")
            ax.set_ylabel(f"{name} relative error [%]")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="best")
            # y-limit: data range, but never wider than +-ymax_cap. Always keep
            # the +-5% band visible by including at least +-(5%) in the window.
            if finite:
                lim = min(ymax_cap, max(abs(min(finite)), abs(max(finite)),
                                        100 * BAND) * 1.1)
            else:
                lim = ymax_cap
            ax.set_ylim(-lim, lim)
        axes[1].set_xlabel("accumulated sampling points (mean per phase)")
        axes[0].set_title(f"Budget config {label}: Fx / Fy rel. error "
                          f"({len(SEEDS)} seeds, mean +-1 std)")
        fname = os.path.join(sub, f"{label}.png")
        _save(fig, fname); figs.append(fig)
    return figs
 
 
def main(max_workers=MAX_WORKERS):
    # Each run_one (one config+seed) is independent and runs its phases serially
    # inside the worker; we only parallelize ACROSS the 25 (config, seed) jobs.
    # Cache is assumed already warm (run once with 1 process first) so workers
    # only READ prior_cache/*.pkl - no write race.
    jobs = [(label, npp, nph, seed)
            for (label, npp, nph) in ALLOCATIONS for seed in SEEDS]
 
    collected = {}            # (label, seed) -> curve
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for fut in as_completed(futs):
            label, seed, curve = fut.result()
            collected[(label, seed)] = curve
 
    # reassemble into results_by_cfg in the original config/seed order
    results_by_cfg = []
    for label, npp, nph in ALLOCATIONS:
        runs = [collected.get((label, s), []) for s in SEEDS]
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
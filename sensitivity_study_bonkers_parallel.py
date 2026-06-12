"""
sensitivity_study_multi_parallel.py
===================================

Combines the two existing studies:

  * sensitivity_study.py        -> arbitrary NAMED configs, each a few overrides
                                   of the initial sampling and/or adaptive config
                                   (w_*, shell, front_frac, fd_step, n_new, ...).
  * sensitivity_study_budget_parallel.py
                                -> runs many SEEDS per config in parallel and
                                   aggregates mean +- 1 std PER PHASE STEP.

So: define any configs you like (like sensitivity_study.py), and each is run
over many seeds with mean/std bands (like the budget study). Each config also
carries its own DRONES_PER_PHASE (n_per_phase) and N_PHASES, so you can mix a
weight sweep with a budget sweep in one run.

It REUSES the aggregation + plotting helpers from
sensitivity_study_budget_parallel.py (imported), so those live in one place.

Prerequisite - WARM CACHE (same as the budget study):
  Grid-free runs only READ the momentum-prior cache (keyed by geometry/v_inf,
  NOT by seed). RUN ONCE WITH 1 PROCESS / MAX_WORKERS=1 first (or a single main.py
  with the same base geometry) so the momentum cache exists; afterwards every
  worker only reads it - concurrent reads are safe.
"""

import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

from GPR import run_gpr
from adaptive import propose_adaptive_points

# Reuse the budget study's COMMON, TRUE_FORCE and ALL aggregation/plotting
# helpers verbatim so there is a single source of truth for them.
import sensitivity_study_budget_parallel as B
from sensitivity_study_budget_parallel import (
    COMMON, TRUE_FORCE, BAND,
    _component, _relerr, _inplane_relerr,
    _aggregate, _band_plot, _save,
)


# ---------------------------------------------------------------------------
# Run knobs
# ---------------------------------------------------------------------------
SEEDS = [7, 42, 67, 420, 1234, 15, 4321, 1324, 4213, 3, 696, 6767]
MAX_WORKERS = 6
PLOT_DIR = "plots_multi"

# Defaults applied to EVERY config unless the config overrides them.
DEFAULT_N_PER_PHASE = 100
DEFAULT_N_PHASES = 4            # phase 0 + (N_PHASES-1) adaptive phases

# Base initial cylinder + base adaptive config (matches the other studies). Each
# named config below overrides a few keys of these, so differences are
# attributable. n_points/n_new default to the config's n_per_phase; seed and
# pool_seed are injected per run.
BASE_INITIAL = dict(
    sample_method="cylinder",
    sample_config={"r_factor": 1.2, "h_factor": 1.5, "tilt_deg": 10,
                   "front_frac": 0.25, "front_half_angle_deg": 45.0,
                   "top_cap": True, "top_cap_frac": 0.2},
)

BASE_ADAPTIVE = dict(
    w_var=0.2, w_grad=0.4, w_vort=0.4,
    pool_size=4000, resample_size=600, score_beta=2.0,
    shell_thick_in=0.20, shell_thick_out=0.20,
    front_frac=0.5, front_half_angle_deg=60.0,
    fd_step=None,                       # fixed grid-free FD step (m); None -> auto
    frac_region1=0.70, frac_region2=0.15, frac_region3=0.15,
    excl_horizontal=1.2, excl_vertical=4.2,
    spread_radius=None,
)


# ---------------------------------------------------------------------------
# Config helpers: build override dicts, exactly like sensitivity_study.py, but
# they are MERGED with the base + seed inside the worker (so they stay seed-free
# and picklable here).
# ---------------------------------------------------------------------------
def init_over(**overrides):
    """Initial-sampling overrides (sample_config keys), e.g. init_over(r_factor=1.4)."""
    return dict(overrides)


def adapt_over(**overrides):
    """Adaptive-config overrides, e.g. adapt_over(w_grad=0.6, w_var=0.2, w_vort=0.2)."""
    return dict(overrides)


def cfg(name, init_overrides=None, adaptive_overrides=None,
        n_per_phase=None, n_phases=None):
    """One named config. n_per_phase / n_phases default to the module defaults.

    Returns a picklable tuple consumed by the worker:
        (name, init_overrides, adaptive_overrides, n_per_phase, n_phases)
    """
    return (
        name,
        dict(init_overrides or {}),
        dict(adaptive_overrides or {}),
        DEFAULT_N_PER_PHASE if n_per_phase is None else n_per_phase,
        DEFAULT_N_PHASES if n_phases is None else n_phases,
    )


# ---------------------------------------------------------------------------
# The configs to compare. Mix any knobs you like; each runs over all SEEDS.
# Uncomment / edit freely. Keep the list short - each config is SEEDS x phases
# runs of run_gpr.
# ---------------------------------------------------------------------------
CONFIGS = [
    cfg("auto"),

    # ---- difficulty-weight sweep ----
    cfg("fd_step=0.1", adaptive_overrides=adapt_over(fd_step=0.1)),
    cfg("fd_step=0.5", adaptive_overrides=adapt_over(fd_step=0.5)),
    cfg("fd_step=1.0", adaptive_overrides=adapt_over(fd_step=1.0)),
    cfg("fd_step=2.0", adaptive_overrides=adapt_over(fd_step=2.0)),
    cfg("fd_step=5.0", adaptive_overrides=adapt_over(fd_step=5.0)),
    cfg("fd_step=10.0", adaptive_overrides=adapt_over(fd_step=10.0)),

    # ---- grid-free FD step sweep ----
    # cfg("fd_step=0.5", adaptive_overrides=adapt_over(fd_step=0.5)),
    # cfg("fd_step=2.0", adaptive_overrides=adapt_over(fd_step=2.0)),
    # cfg("fd_step=auto", adaptive_overrides=adapt_over(fd_step=None)),

    # ---- shell thickness ----
    # cfg("big shell",   adaptive_overrides=adapt_over(shell_thick_in=0.3, shell_thick_out=0.3)),
    # cfg("small shell", adaptive_overrides=adapt_over(shell_thick_in=0.1, shell_thick_out=0.1)),

    # ---- initial-sampling geometry ----
    # cfg("r_factor=1.4", init_overrides=init_over(r_factor=1.4)),

    # ---- per-config budget (drones/phase and #phases can differ per config) ----
    # cfg("100x4", n_per_phase=100, n_phases=4),
    # cfg("160x3", n_per_phase=160, n_phases=3),
]


# ---------------------------------------------------------------------------
# Build the concrete run configs (merge base + overrides + seed) inside worker
# ---------------------------------------------------------------------------
def _make_initial(init_overrides, n_per_phase, seed):
    init = {k: (dict(v) if isinstance(v, dict) else v)
            for k, v in BASE_INITIAL.items()}
    sc = init["sample_config"]
    sc.update(init_overrides)
    sc["n_points"] = init_overrides.get("n_points", n_per_phase)
    # carve a top-cap fraction out of the phase-0 budget (>=20 drones or 20%),
    # matching the budget study's behaviour.
    sc["top_cap_frac"] = max(20.0 / max(n_per_phase, 1), 0.20)
    sc["seed"] = seed
    return init


def _make_adaptive(adaptive_overrides, n_per_phase, seed):
    c = dict(BASE_ADAPTIVE)
    c.update(adaptive_overrides)
    c["n_new"] = adaptive_overrides.get("n_new", n_per_phase)
    c["seed"] = seed
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


# ---------------------------------------------------------------------------
# One (config, seed) run: phase 0 + (n_phases-1) adaptive phases, sequential.
# Returns list of (n_accumulated, force_vec). Mirrors the budget study's run_one
# but with arbitrary init/adaptive overrides.
# ---------------------------------------------------------------------------
def run_one(name, init_overrides, adaptive_overrides, n_per_phase, n_phases, seed):
    print(f"\n########## {name}  seed={seed} ##########", flush=True)
    initial = _make_initial(init_overrides, n_per_phase, seed)
    adaptive_cfg = _make_adaptive(adaptive_overrides, n_per_phase, seed)

    result = run_gpr(**COMMON, **initial)
    cyl = result.get("cylinder_geom")
    if cyl is None:
        raise RuntimeError(f"[{name}/{seed}] phase 0 produced no cylinder_geom.")
    accumulated = np.asarray(result["training_coords"], float)
    curve = [(len(accumulated), result["metrics"].get("force_vec"))]
    fv = result["metrics"].get("force_vec")
    if fv is not None:
        ex = abs(fv[0] - TRUE_FORCE[0]) / abs(TRUE_FORCE[0])
        ey = abs(fv[1] - TRUE_FORCE[1]) / abs(TRUE_FORCE[1])
        print(f"[{name}/{seed}] phase 0  {len(accumulated)} pts  "
              f"relx={ex:.4g}  rely={ey:.4g}", flush=True)

    for phase in range(1, n_phases):
        result["cylinder_geom"] = cyl
        phase_cfg = dict(adaptive_cfg)
        phase_cfg["pool_seed"] = seed + phase     # fresh candidate cloud per phase
        new_pts = propose_adaptive_points(
            result, previous_coords=accumulated,
            adaptive_config=phase_cfg, verbose=False)
        if len(new_pts) == 0:
            print(f"[{name}/{seed}] phase {phase}: no new points; stopping early.",
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
            print(f"[{name}/{seed}] phase {phase}  {len(accumulated)} pts  "
                  f"relx={ex:.4g}  rely={ey:.4g}", flush=True)

    _free_heavy(result)
    gc.collect()
    return curve


def _job(args):
    """Picklable worker: one (config, seed) job -> (name, seed, curve)."""
    name, init_o, adapt_o, npp, nph, seed = args
    try:
        return (name, seed, run_one(name, init_o, adapt_o, npp, nph, seed))
    except Exception as e:
        print(f"[{name}/{seed}] FAILED: {e!r}", flush=True)
        return (name, seed, [])


# ---------------------------------------------------------------------------
# Plot/aggregate. results_by_cfg uses the SAME shape the budget helpers expect:
#   list of (label, runs, n_per_phase, n_phases)
# where runs is a list (over seeds) of curves. So we can call B's figure makers.
# ---------------------------------------------------------------------------
def _make_figs(results_by_cfg):
    os.makedirs(PLOT_DIR, exist_ok=True)
    # point the budget module's PLOT_DIR at ours so its _save paths land here if
    # any helper builds its own filename (we pass explicit names below anyway).
    B.PLOT_DIR = PLOT_DIR
    figs = []

    # absolute Fx / Fy bands
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    _band_plot(axes[0], results_by_cfg, lambda fv: _component(fv, 0),
               "Fx [N]", "Multi-config: Fx / Fy convergence (mean +-1 std over seeds)",
               hline=TRUE_FORCE[0], band_abs=TRUE_FORCE[0])
    _band_plot(axes[1], results_by_cfg, lambda fv: _component(fv, 1),
               "Fy [N]", "", hline=TRUE_FORCE[1], band_abs=TRUE_FORCE[1])
    _save(fig, os.path.join(PLOT_DIR, "multi_fx_fy_band.png")); figs.append(fig)

    # signed rel-error Fx / Fy bands
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    _band_plot(axes[0], results_by_cfg, lambda fv: _relerr(fv, 0),
               "Fx rel. error [%]",
               "Multi-config: Fx / Fy relative error (mean +-1 std)",
               hline=0.0, band_rel_true=True, ymax_cap=50.0)
    _band_plot(axes[1], results_by_cfg, lambda fv: _relerr(fv, 1),
               "Fy rel. error [%]", "", hline=0.0, band_rel_true=True, ymax_cap=50.0)
    _save(fig, os.path.join(PLOT_DIR, "multi_fx_fy_relerr_band.png")); figs.append(fig)

    # in-plane |Fx,Fy| rel error
    fig, ax = plt.subplots(figsize=(10, 5))
    _band_plot(ax, results_by_cfg, _inplane_relerr,
               "in-plane (Fx,Fy) rel error [%]",
               "Multi-config: in-plane relative error (mean +-1 std)", hline=0.0)
    ax.axhspan(0.0, 100 * BAND, color="0.5", alpha=0.12, label="+-5% band")
    top = ax.get_ylim()[1]
    ax.set_ylim(0.0, min(50.0, max(top, 100 * BAND * 1.1)))
    _save(fig, os.path.join(PLOT_DIR, "multi_relerr_inplane_band.png")); figs.append(fig)
    return figs


def _print_summary(results_by_cfg):
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


def main(max_workers=MAX_WORKERS):
    jobs = [(name, io, ao, npp, nph, seed)
            for (name, io, ao, npp, nph) in CONFIGS for seed in SEEDS]

    collected = {}            # (name, seed) -> curve
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for fut in as_completed(futs):
            name, seed, curve = fut.result()
            collected[(name, seed)] = curve

    # reassemble in config order, runs aligned to SEEDS order
    results_by_cfg = []
    for (name, io, ao, npp, nph) in CONFIGS:
        runs = [collected.get((name, s), []) for s in SEEDS]
        results_by_cfg.append((name, runs, npp, nph))

    # final-phase in-plane error per seed (quick scan)
    print("\n\n=== final-phase in-plane (Fx,Fy) relative error [%], per seed ===")
    for name, runs, npp, nph in results_by_cfg:
        cells = []
        for seed, c in zip(SEEDS, runs):
            cells.append(f"s{seed}:fail" if not c
                         else f"s{seed}:{_inplane_relerr(c[-1][1]):.2f}")
        print(f"  {name:16s}: " + "  ".join(cells))

    _print_summary(results_by_cfg)

    os.makedirs(PLOT_DIR, exist_ok=True)
    _make_figs(results_by_cfg)
    plt.show()
    return results_by_cfg


if __name__ == "__main__":
    main()

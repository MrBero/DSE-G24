
import os

# ---------------------------------------------------------------------------
# Thread budget (set BEFORE numpy/JAX/BLAS import so they honor it).
# 7800X3D = 8 physical cores / 16 SMT threads. With many independent
# (config, seed) jobs the work is embarrassingly parallel ACROSS jobs, so
# throughput comes from running many serial workers, not from threading each
# one. We pin each worker to THREADS_PER_WORKER and run MAX_WORKERS of them.
#   8 workers x 1 thread  = 8 (one per physical core)  <- default, best for sweeps
#   4 workers x 2 threads = 8                            <- try if single runs feel slow
# These env vars cap NumPy's BLAS (OpenBLAS/MKL), OpenMP, and JAX/XLA. Julia is
# pinned separately via JULIA_NUM_THREADS (the FP.jl subprocess inherits it).
THREADS_PER_WORKER = 2
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "JULIA_NUM_THREADS"):
    os.environ.setdefault(_v, str(THREADS_PER_WORKER))
# Keep XLA single-threaded per worker too (JAX is forced onto CPU in GPR.py).
os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={THREADS_PER_WORKER}")

import gc
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

import trimesh
from flowpanelwrapper import FLOWPanelSolver
from GPR import run_gpr
from adaptive import propose_adaptive_points


# ---------------------------------------------------------------------------
# Shared run settings (identical to the other studies so only sampling changes)
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

# ---- reference forces [Fx, Fy, Fz] in N ----
# TRUE_FORCE: the CFD/ANSYS-reported force (physical truth).
TRUE_FORCE = np.array([155433.0, 208647.0, 72586.0])

# MOMENTUM_FORCE: momentum integral evaluated on the interpolated ground-truth
# field with many surface points (the integral's own floor). Set the values when
# you have them; leave as None to skip the second (vs-momentum) comparison.
# Example once known:  MOMENTUM_FORCE = np.array([Fx_mom, Fy_mom, Fz_mom])
MOMENTUM_FORCE = np.array([155433.0, 214609.0, 72586.0])


# ---------------------------------------------------------------------------
# Run knobs
# ---------------------------------------------------------------------------
import random
def generate_seeds(n=4, min_gap=7, lo=0, hi=100000, rng_seed=None):
    rng = random.Random(rng_seed)
    slots = (hi - lo) // min_gap + 1
    if slots < n:
        raise ValueError("range too small for n seeds at this spacing")
    chosen = rng.sample(range(slots), n)        # distinct slot indices
    return sorted(lo + s * min_gap for s in chosen)
SEEDS = generate_seeds(rng_seed=67)
print(SEEDS)
# SEEDS = [7, 42, 67, 420]#, 1234, 15, 4321, 1324, 4213, 3, 696, 6767, 89403, 132, 432, 594]
MAX_WORKERS = 4   # 8 workers x 1 thread each = 8 physical cores (see THREADS_PER_WORKER)
PLOT_DIR = "plots_400"
BAND = 0.05                          # +-5% acceptance band drawn on plots

# Defaults applied to EVERY config unless the config overrides them.
DEFAULT_N_PER_PHASE = 80
DEFAULT_N_PHASES = 2                 # phase 0 + (N_PHASES-1) adaptive phases

# Base initial cylinder + base adaptive config (matches the other studies).
BASE_INITIAL = dict(
    sample_method="cylinder",
    sample_config={"r_factor": 1.2, "h_factor": 1.5, "tilt_deg": 10,
                   "front_frac": 0.25, "front_half_angle_deg": 45.0,
                   "top_cap": False, "top_cap_frac": 0.2},
)

BASE_ADAPTIVE = dict(
    w_var=0.2, w_grad=0.4, w_vort=0.4,
    pool_size=5000, resample_size=800, score_beta=0.0,
    shell_thick_in=0.20, shell_thick_out=0.20,
    front_frac=0.5, front_half_angle_deg=60.0,
    frac_region1=0.70, frac_region2=0.15, frac_region3=0.15,
    excl_horizontal=1.2, excl_vertical=4.2,
    spread_radius=None,
    fd_step=[1.0,1.0,1.0]
)


# ---------------------------------------------------------------------------
# Config helpers: build override dicts; merged with base + seed in the worker.
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
    Returns a picklable tuple consumed by the worker."""
    return (
        name,
        dict(init_overrides or {}),
        dict(adaptive_overrides or {}),
        DEFAULT_N_PER_PHASE if n_per_phase is None else n_per_phase,
        DEFAULT_N_PHASES if n_phases is None else n_phases,
    )


# ---------------------------------------------------------------------------
# The configs to compare. Each runs over all SEEDS.
# ---------------------------------------------------------------------------
CONFIGS = [
        cfg("80",  n_per_phase=80),
        cfg("160",  n_per_phase=160),
        cfg("240",  n_per_phase=240),
        cfg("320",  n_per_phase=320),
        cfg("400",  n_per_phase=400),
        cfg("480",  n_per_phase=480),
        cfg("560",  n_per_phase=560),
]


# ---------------------------------------------------------------------------
# Build the concrete run configs (merge base + overrides + seed) in the worker
# ---------------------------------------------------------------------------
def _make_initial(init_overrides, n_per_phase, seed):
    init = {k: (dict(v) if isinstance(v, dict) else v)
            for k, v in BASE_INITIAL.items()}
    sc = init["sample_config"]
    sc.update(init_overrides)
    sc["n_points"] = init_overrides.get("n_points", 1)
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
# Returns list of (n_accumulated, force_vec).
# ---------------------------------------------------------------------------
def run_one(name, init_overrides, adaptive_overrides, n_per_phase, n_phases, seed):
    print(f"\n########## {name}  seed={seed} ##########", flush=True)
    initial = _make_initial(init_overrides, n_per_phase, seed)
    adaptive_cfg = _make_adaptive(adaptive_overrides, n_per_phase, seed)

    # One Julia solver per job, reused across this job's phases (mesh + v_inf are
    # constant across all configs/seeds). Saves the per-phase Julia cold start.
    _mesh = trimesh.load_mesh(COMMON["stl_filepath"])
    if COMMON["stl_scale"] != 1.0:
        _mesh.apply_scale(COMMON["stl_scale"])
    solver = FLOWPanelSolver(_mesh, COMMON["v_inf"], julia_script="FP.jl",
                             julia_bin="julia", verbose=False)
    try:
        return _run_one_impl(name, initial, adaptive_cfg, n_phases, seed, solver)
    finally:
        try:
            solver.close()
        except Exception:
            pass


def _run_one_impl(name, initial, adaptive_cfg, n_phases, seed, solver):
    result = run_gpr(**COMMON, **initial, solver=solver)
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
                         samples=accumulated, cylinder_geom_override=cyl,
                         solver=solver)
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


# ===========================================================================
# Aggregation helpers. Error metrics take an explicit reference force `ref`, so
# the same GPR curves can be scored against TRUE_FORCE or MOMENTUM_FORCE.
# ===========================================================================
def _component(fv, comp):
    return np.nan if fv is None else float(np.asarray(fv, float)[comp])


def _relerr(fv, comp, ref):
    """Signed relative error [%] of component `comp` against reference `ref`."""
    if fv is None:
        return np.nan
    return 100.0 * (np.asarray(fv, float)[comp] - ref[comp]) / abs(ref[comp])


def _inplane_relerr(fv, ref):
    """In-plane (Fx,Fy) relative error magnitude [%] against reference `ref`."""
    if fv is None:
        return np.nan
    d = np.asarray(fv, float)[:2] - ref[:2]
    return 100.0 * np.linalg.norm(d) / np.linalg.norm(ref[:2])


def _aggregate(all_runs, n_per_phase, n_phases, fn):
    """Average fn(force_vec) across seeds at each PHASE STEP (0..n_phases-1).
    Returns (x, mean, std, x_std). Each seed's curve has len == n_phases, so
    phase step k is curve[k], no interpolation. Early-stopped seeds contribute
    nan for missing tail phases (dropped by nanmean)."""
    runs = [c for c in all_runs if c]
    n_steps = n_phases
    counts = np.full((len(runs), n_steps), np.nan)
    vals = np.full((len(runs), n_steps), np.nan)
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


# ===========================================================================
# Plotting helpers
# ===========================================================================
def _band_plot(ax, results_by_cfg, fn, ylabel, title, hline=None, band_abs=None,
               band_rel_true=None, x_err=False, ymax_cap=None, fill_style="dashed"):
    """Mean line +-1std per config. fn -> scalar from force_vec."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_by_cfg)))
    finite = []
    for (label, runs, npp, nph), color in zip(results_by_cfg, colors):
        x, mean, std, x_std = _aggregate(runs, npp, nph, fn)
        ax.plot(x, mean, "-o", color=color, label=label, markersize=4)
        if fill_style == "fill":
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
        else:
            ax.plot(x, mean - std, ls="--", color=color, lw=0.9, alpha=0.4)
            ax.plot(x, mean + std, ls="--", color=color, lw=0.9, alpha=0.4)
        finite += [v for v in (mean - std) if np.isfinite(v)]
        finite += [v for v in (mean + std) if np.isfinite(v)]
        if x_err:
            ax.errorbar(x, mean, xerr=x_std, fmt="none", ecolor=color,
                        alpha=0.5, capsize=2)
    if hline is not None:
        ax.axhline(hline, ls="--", color="0.4")
    if band_abs is not None:           # absolute force: +-5% band around the ref
        ax.axhspan(band_abs * (1 - BAND), band_abs * (1 + BAND),
                   color="0.5", alpha=0.12, label="+-5% band")
    if band_rel_true is not None:      # rel-error plot: +-5% band around 0
        ax.axhspan(-100 * BAND, 100 * BAND, color="0.5", alpha=0.12,
                   label="+-5% band")
    if ymax_cap is not None:
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


def _make_per_config_figs(results_by_cfg, ref, ref_name, out_dir, ymax_cap=50.0):
    """One detailed figure per config: Fx and Fy relative-error convergence with
    band, plus every individual seed line faint underneath. Errors are against
    `ref` (labelled by ref_name); files land in out_dir/per_config_<ref_name>/."""
    sub = os.path.join(out_dir, f"per_config_{ref_name}")
    os.makedirs(sub, exist_ok=True)
    figs = []
    for label, runs, npp, nph in results_by_cfg:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        for comp, (ax, name) in enumerate(zip(axes, ("Fx", "Fy"))):
            finite = []
            for c in runs:
                if not c:
                    continue
                ns = [p[0] for p in c]
                vs = [_relerr(p[1], comp, ref) for p in c]
                ax.plot(ns, vs, "-", color="0.7", lw=0.8, alpha=0.6)
                finite += [v for v in vs if np.isfinite(v)]
            x, mean, std, x_std = _aggregate(
                runs, npp, nph, lambda fv, cc=comp: _relerr(fv, cc, ref))
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
            if finite:
                lim = min(ymax_cap, max(abs(min(finite)), abs(max(finite)),
                                        100 * BAND) * 1.1)
            else:
                lim = ymax_cap
            ax.set_ylim(-lim, lim)
        axes[1].set_xlabel("accumulated sampling points (mean per phase)")
        axes[0].set_title(f"{label}: Fx / Fy rel. error vs {ref_name} "
                          f"({len(SEEDS)} seeds, mean +-1 std)")
        fname = os.path.join(sub, f"{label}.png")
        _save(fig, fname); figs.append(fig)
    return figs


def _make_figs_for_ref(results_by_cfg, ref, ref_name):
    """Emit the full figure set for one reference force (ref, labelled ref_name).
    Files are suffixed _vs_<ref_name> so the truth and momentum sets don't clash."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    figs = []
    suff = f"_vs_{ref_name}"

    # absolute Fx / Fy bands (band drawn around this reference)
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    _band_plot(axes[0], results_by_cfg, lambda fv: _component(fv, 0),
               "Fx [N]",
               f"Fx / Fy convergence vs {ref_name} (mean +-1 std over seeds)",
               hline=ref[0], band_abs=ref[0])
    _band_plot(axes[1], results_by_cfg, lambda fv: _component(fv, 1),
               "Fy [N]", "", hline=ref[1], band_abs=ref[1])
    _save(fig, os.path.join(PLOT_DIR, f"fx_fy_band{suff}.png")); figs.append(fig)

    # signed rel-error Fx / Fy bands (against this reference)
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    _band_plot(axes[0], results_by_cfg, lambda fv: _relerr(fv, 0, ref),
               "Fx rel. error [%]",
               f"Fx / Fy relative error vs {ref_name} (mean +-1 std)",
               hline=0.0, band_rel_true=True, ymax_cap=50.0)
    _band_plot(axes[1], results_by_cfg, lambda fv: _relerr(fv, 1, ref),
               "Fy rel. error [%]", "", hline=0.0, band_rel_true=True, ymax_cap=50.0)
    _save(fig, os.path.join(PLOT_DIR, f"fx_fy_relerr_band{suff}.png")); figs.append(fig)

    # in-plane |Fx,Fy| rel error (against this reference)
    fig, ax = plt.subplots(figsize=(10, 5))
    _band_plot(ax, results_by_cfg, lambda fv: _inplane_relerr(fv, ref),
               "in-plane (Fx,Fy) rel error [%]",
               f"in-plane relative error vs {ref_name} (mean +-1 std)", hline=0.0)
    ax.axhspan(0.0, 100 * BAND, color="0.5", alpha=0.12, label="+-5% band")
    top = ax.get_ylim()[1]
    ax.set_ylim(0.0, min(50.0, max(top, 100 * BAND * 1.1)))
    _save(fig, os.path.join(PLOT_DIR, f"relerr_inplane_band{suff}.png")); figs.append(fig)

    # per-config detail
    figs += _make_per_config_figs(results_by_cfg, ref, ref_name, PLOT_DIR)
    return figs


def _print_summary_for_ref(results_by_cfg, ref, ref_name):
    print(f"\n\n=== rel. error [%] vs {ref_name}, mean +- std over seeds, per phase ===")
    for label, runs, npp, nph in results_by_cfg:
        x, mx, sx, xstd = _aggregate(runs, npp, nph, lambda fv: _relerr(fv, 0, ref))
        _, my, sy, _ = _aggregate(runs, npp, nph, lambda fv: _relerr(fv, 1, ref))
        _, mip, sip, _ = _aggregate(runs, npp, nph, lambda fv: _inplane_relerr(fv, ref))
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


def _references():
    """List of (ref_force, ref_name) to score against. Always includes truth;
    includes momentum only if MOMENTUM_FORCE is set."""
    refs = [(np.asarray(TRUE_FORCE, float), "truth")]
    if MOMENTUM_FORCE is not None:
        refs.append((np.asarray(MOMENTUM_FORCE, float), "momentum"))
    return refs


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

    os.makedirs(PLOT_DIR, exist_ok=True)

    # For each reference (truth, and momentum if set): quick per-seed scan,
    # numeric summary, and the full figure set.
    for ref, ref_name in _references():
        print(f"\n\n=== final-phase in-plane (Fx,Fy) rel error [%] vs {ref_name}, per seed ===")
        for name, runs, npp, nph in results_by_cfg:
            cells = []
            for seed, c in zip(SEEDS, runs):
                cells.append(f"s{seed}:fail" if not c
                             else f"s{seed}:{_inplane_relerr(c[-1][1], ref):.2f}")
            print(f"  {name:16s}: " + "  ".join(cells))
        _print_summary_for_ref(results_by_cfg, ref, ref_name)
        _make_figs_for_ref(results_by_cfg, ref, ref_name)

    plt.show()
    return results_by_cfg


if __name__ == "__main__":
    main()

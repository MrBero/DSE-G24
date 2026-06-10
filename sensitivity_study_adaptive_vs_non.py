"""
adaptive_vs_non.py
==================

Standalone driver (imports run_gpr / adaptive, modifies nothing) that isolates
the two independent design axes of the adaptive sampler:

    Axis 1 - scored vs flat : does grad/var/vort weighting beat uniform LHS
                              over the SAME thick-shell candidate distribution?
    Axis 2 - iterated vs one-shot : does refilling the difficulty field over
                              several phases beat one big draw at equal budget?

A real adaptive run is (scored, iterated). We compare three arms, ALL drawing
from the identical thick-cylinder shell geometry (same shell_thick_*, region
fracs, front bias) so the ONLY differences are the two axes above:

    Arm A  scored  + iterated   -> the full method            (score_beta=2, N phases)
    Arm B  flat    + iterated   -> isolates SCORING vs A       (score_beta=0, N phases)
    Arm C  flat    + one-shot   -> isolates PHASING vs B       (score_beta=0, 1 phase)

  A vs B : does the weighting help, phasing held equal.
  B vs C : does iterating help/hurt, scoring held off.

  (scored, one-shot) is intentionally absent - one-shot has no informative
  posterior to score against, so it is not a meaningful arm.

WHY score_beta=0 == "flat LHS over the shell":
  in adaptive.propose_adaptive_points, weights = clip(score,0,None) ** score_beta.
  With score_beta=0 every candidate's weight is x**0 == 1, so the importance
  resample becomes a uniform draw over the shell pool and the greedy seeding is
  pure farthest-point spread. Same geometry as the scored arm, no weighting.

WHY 3 seeds:
  earlier runs showed ~10% Fy swing between two scoring-OFF draws (pure LHS
  luck). A single seed cannot resolve a scoring effect smaller than that, so we
  run seeds 7/8/9 per arm and compare arm MEANS with spread. A gap inside the
  combined spread = "no detectable effect", not "effect".

WHY cap at 3 phases (for the iterated arms):
  the on-face band saturates against the 4.2 m drone-exclusion ellipsoid by
  ~phase 4 (face placement drops to 0); past that, scoring has nothing left to
  bias on the face and later phases only dilute the comparison. 3 phases keeps
  the test in the regime where scoring CAN act.

MATCHED COUNT (important):
  exclusion drops vary per draw, so arms do NOT accumulate identical totals.
  Arm C's n_new is set generously so its single phase lands near where A/B end;
  the comparison is read at MATCHED DRONE COUNT off the x-axis, never at matched
  phase index. The plot makes this visual.

Outputs:
  - console table: per-config Fx/Fy error vs #drones each phase
  - per-arm mean +/- spread Fy error vs drones (interpolated to a common grid)
  - plots/adaptive_vs_non.pdf : Fx and Fy convergence, one line per config,
    with a +/-5% acceptance band around the true force
  - an on-screen interactive window (in addition to the PDF), if a display is
    available; harmless no-op when headless
"""

import os
import gc
import numpy as np
import matplotlib

# We want BOTH a saved PDF (always) and an on-screen window (when possible).
# savefig works under any backend, so we prefer an interactive backend if one is
# available and only fall back to headless Agg when no display exists. This lets
# the single figure be both saved and shown without re-rendering.
_INTERACTIVE = True
try:
    matplotlib.use("TkAgg")            # common, ships with most python installs
except Exception:
    try:
        matplotlib.use("QtAgg")        # fall back to Qt if Tk isn't present
    except Exception:
        matplotlib.use("Agg")          # headless: PDF only, no window
        _INTERACTIVE = False
import matplotlib.pyplot as plt

from GPR import run_gpr
from adaptive import propose_adaptive_points


# ---------------------------------------------------------------------------
# Shared settings - copied from the latest main.py so geometry matches exactly
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

# Acceptance band shown on the convergence plot (fraction of true component).
ERR_BAND = 0.05          # +/- 5%

# Default phase counts. Iterated arms (A,B) use N_PHASES_ITER; the one-shot arm
# (C) overrides this to 1 via its per-config n_phases (see CONFIGS / run_one).
N_PHASES_ITER = 5
DRONES_PER_PHASE = 80    # phase 0 and every iterated adaptive phase add this many

# Phase-0 cylinder sampling - matches latest main.py INITIAL_SAMPLING.
BASE_INITIAL = dict(
    sample_method="cylinder",
    sample_config={"r_factor": 1.2, "h_factor": 1.5, "tilt_deg": 10,
                   "n_points": DRONES_PER_PHASE,
                   "front_frac": 0.5, "front_half_angle_deg": 45.0,
                   "seed": 7},
)

# Base adaptive config - matches latest main.py ADAPTIVE_CFG. Each named config
# below is this dict with a few keys overridden, so differences are attributable.
BASE_ADAPTIVE = dict(
    w_var=0.2, w_grad=0.4, w_vort=0.4,
    pool_size=4000, resample_size=600, score_beta=2.0,
    shell_thick_in=0.50, shell_thick_out=0.30,
    front_frac=0.5, front_half_angle_deg=60.0,
    n_new=DRONES_PER_PHASE,
    frac_region1=0.6, frac_region2=0.25, frac_region3=0.15,  # face / inner / outer
    excl_horizontal=4.2, excl_vertical=4.2,
    spread_radius=None,
    seed=7,
)


def _cfg(**overrides):
    c = dict(BASE_ADAPTIVE); c.update(overrides); return c


def _init(**sample_overrides):
    init = {k: (dict(v) if isinstance(v, dict) else v)
            for k, v in BASE_INITIAL.items()}
    init["sample_config"].update(sample_overrides)
    return init


# ---------------------------------------------------------------------------
# The arms. (name, initial_sampling, adaptive_config, n_phases)
# A/B iterate (3 phases x 80); C is one-shot (1 phase, big n_new to match count).
# 3 seeds per arm so we can compare MEANS with spread, not single draws.
# ---------------------------------------------------------------------------
CONFIGS = [
    # Arm A: full method - scored + iterated
    ("scored_iter_s7",   _init(seed=7), _cfg(score_beta=2.0, n_new=80,  seed=7), N_PHASES_ITER),
    ("scored_iter_s8",   _init(seed=8), _cfg(score_beta=2.0, n_new=80,  seed=8), N_PHASES_ITER),
    ("scored_iter_s9",   _init(seed=9), _cfg(score_beta=2.0, n_new=80,  seed=9), N_PHASES_ITER),
    # Arm B: flat + iterated  (isolates scoring vs A)
    ("flat_iter_s7",     _init(seed=7), _cfg(score_beta=0.0, n_new=80,  seed=7), N_PHASES_ITER),
    ("flat_iter_s8",     _init(seed=8), _cfg(score_beta=0.0, n_new=80,  seed=8), N_PHASES_ITER),
    ("flat_iter_s9",     _init(seed=9), _cfg(score_beta=0.0, n_new=80,  seed=9), N_PHASES_ITER),
    # Arm C: flat + one-shot  (isolates phasing vs B). n_new generous so the
    # single phase lands near where A/B end (~240); compared at matched count.
    ("flat_oneshot_s7",  _init(seed=7), _cfg(score_beta=0.0, n_new=N_PHASES_ITER*80, seed=7), 1),
    ("flat_oneshot_s8",  _init(seed=8), _cfg(score_beta=0.0, n_new=N_PHASES_ITER*80, seed=8), 1),
    ("flat_oneshot_s9",  _init(seed=9), _cfg(score_beta=0.0, n_new=N_PHASES_ITER*80, seed=9), 1),
]

# Map each config name to the arm it belongs to (for mean +/- spread grouping).
def _arm_of(name):
    if name.startswith("scored_iter"):
        return "A: scored+iter"
    if name.startswith("flat_iter"):
        return "B: flat+iter"
    if name.startswith("flat_oneshot"):
        return "C: flat+oneshot"
    return "other"

ARM_COLORS = {
    "A: scored+iter":   "tab:blue",
    "B: flat+iter":     "tab:orange",
    "C: flat+oneshot":  "tab:green",
}


def _free_heavy(res):
    """Drop the big arrays + CFD sampler closure from a finished result so the
    21M-point interpolator it pins gets reclaimed before the next run_gpr builds
    a fresh one. Without this, the many run_gpr calls across a study accumulate
    enough interpolators to exhaust memory."""
    if res is None:
        return
    for k in ("sample_dat_shi", "test_points", "GPR_posterior", "GPR_variances",
              "means_tests", "cfd_test_vels", "pressure_posterior", "momentum"):
        res.pop(k, None)


def run_one(name, initial_sampling, adaptive_cfg, n_phases):
    """Run phase 0 + n_phases adaptive phases for one config.
    Returns list of (n_drones, force_vec) including phase 0."""
    print(f"\n########## CONFIG: {name}  (phases={n_phases}, "
          f"beta={adaptive_cfg['score_beta']}, n_new={adaptive_cfg['n_new']}) ##########")
    result = run_gpr(**COMMON, **initial_sampling)
    cyl = result.get("cylinder_geom")
    if cyl is None:
        raise RuntimeError(f"[{name}] phase 0 produced no cylinder_geom.")
    accumulated = np.asarray(result["training_coords"], float)
    curve = [(len(accumulated), result["metrics"].get("force_vec"))]

    for phase in range(1, n_phases + 1):
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


def _relc(fv, i):
    """Relative error of component i vs TRUE_FORCE, or nan."""
    if fv is None:
        return np.nan
    return abs(np.asarray(fv, float)[i] - TRUE_FORCE[i]) / abs(TRUE_FORCE[i])


def _fxfy_err(force_vec):
    """In-plane (Fx,Fy) relative error magnitude vs TRUE_FORCE, ignoring Fz."""
    if force_vec is None:
        return np.nan
    fv = np.asarray(force_vec, float)[:2]
    tf = TRUE_FORCE[:2]
    return np.linalg.norm(fv - tf) / np.linalg.norm(tf)


def _arm_mean_curve(curves, comp_idx):
    """Given a list of per-seed curves [(n, fv), ...] for one arm, interpolate
    each seed's relative error of component comp_idx onto a common drone-count
    grid and return (grid, mean, std). Lets us plot mean +/- spread on aligned x
    even though seeds land at slightly different accumulated counts."""
    # collect each seed's (n, rel_err) with finite values
    seqs = []
    for curve in curves:
        ns = np.array([c[0] for c in curve], float)
        es = np.array([_relc(c[1], comp_idx) for c in curve], float)
        ok = np.isfinite(ns) & np.isfinite(es)
        if ok.sum() >= 2:
            seqs.append((ns[ok], es[ok]))
    if not seqs:
        return None, None, None
    lo = max(s[0].min() for s in seqs)
    hi = min(s[0].max() for s in seqs)
    if not (hi > lo):
        return None, None, None
    grid = np.linspace(lo, hi, 25)
    stacks = [np.interp(grid, s[0], s[1]) for s in seqs]
    arr = np.vstack(stacks)
    return grid, arr.mean(axis=0), arr.std(axis=0)


def main():
    all_curves = {}
    for name, init, cfg, n_phases in CONFIGS:
        try:
            all_curves[name] = run_one(name, init, cfg, n_phases)
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
        cells = [f"{n_d}:{_fxfy_err(fv):.3g}" for n_d, fv in curve]
        print(f"  {name:18s}: " + "  ".join(cells))

    # ---- per-arm mean +/- spread of Fy error at matched drone counts ----
    arms = {}
    for name, curve in all_curves.items():
        if curve:
            arms.setdefault(_arm_of(name), []).append(curve)
    print("\n=== Fy relative error: per-arm mean +/- std (interpolated to common grid) ===")
    for arm, curves in arms.items():
        grid, mean, std = _arm_mean_curve(curves, comp_idx=1)
        if grid is None:
            print(f"  {arm:18s}: (insufficient data)")
            continue
        lo_n, hi_n = int(grid[0]), int(grid[-1])
        print(f"  {arm:18s}: at ~{hi_n} drones  Fy rel = "
              f"{mean[-1]:.4g} +/- {std[-1]:.2g}   "
              f"(range {lo_n}-{hi_n} drones, {len(curves)} seeds)")

    # ---- convergence plot: Fx and Fy, one line per config, +/-ERR_BAND band ----
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    for name, curve in all_curves.items():
        if not curve:
            continue
        color = ARM_COLORS.get(_arm_of(name), None)
        ns = [c[0] for c in curve]
        fx = [c[1][0] if c[1] is not None else np.nan for c in curve]
        fy = [c[1][1] if c[1] is not None else np.nan for c in curve]
        axes[0].plot(ns, fx, "-o", color=color, label=name, markersize=4, alpha=0.85)
        axes[1].plot(ns, fy, "-o", color=color, label=name, markersize=4, alpha=0.85)

    # true lines + +/- ERR_BAND acceptance bands
    for ax, comp in zip(axes, (0, 1)):
        tval = TRUE_FORCE[comp]
        ax.axhline(tval, ls="--", color="0.4",
                   label=f"true {'Fx' if comp == 0 else 'Fy'}")
        ax.axhspan(tval * (1 - ERR_BAND), tval * (1 + ERR_BAND),
                   color="0.5", alpha=0.15,
                   label=f"+/-{int(ERR_BAND*100)}% band")

    axes[0].set_ylabel("Fx [N]"); axes[1].set_ylabel("Fy [N]")
    axes[1].set_xlabel("number of training drones")
    axes[0].set_title("Adaptive vs non-adaptive: Fx / Fy convergence "
                      f"(shaded = +/-{int(ERR_BAND*100)}% of true)")
    for ax in axes:
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out = os.path.join("plots", "adaptive_vs_non.pdf")
    fig.savefig(out)
    print(f"\nsaved {out}")

    # ---- also pop an interactive window, if a display is available ----
    # Backend was chosen at import time; only call show() if it's interactive.
    if _INTERACTIVE:
        try:
            plt.show()
        except Exception as e:
            print(f"[plot] interactive display unavailable ({e!r}); PDF saved only.")
    else:
        print("[plot] headless backend; PDF saved only (no on-screen window).")

    return all_curves


if __name__ == "__main__":
    main()

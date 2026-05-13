"""
forces.py - Summarise final aerodynamic forces from CFD simulation runs and
produce per-component overview plots.

Pipeline (run once, automatically scales with whatever is on disk):
  1. Walks every `sims_(vel_mean_*_aoa_mean_*)` folder inside
     `forces_visualisation/`.
  2. For every sub-case it reads the last iteration of `forces.txt` and
     extracts the final fx, fy.
  3. Writes one summary CSV per sims-folder into `forces_output/`, named
     `forces_<meanUinlet>_<meanAOA>.csv`. Each CSV has 6 rows:
        fx  lowest / most likely / highest
        fy  lowest / most likely / highest
     with the (U_inlet, AoA) situation that produced each bound.
  4. Aggregates every scenario into two forest-style plots saved to
     `forces_output/`:
        forces_fx.png   - one row per scenario, fx in N
        forces_fy.png   - one row per scenario, fy in N
     Each row shows the [lowest, highest] whisker, a marker at the
     'most likely' (mean U, mean AoA) value, and a different-coloured
     marker at the user-set 'true value' reference.

The plots scale automatically with the number of sims-folders present, so
dropping in more `sims_(...)` zips and re-running is all that's needed.

Paths are resolved relative to this script's own location, so the project
is portable across machines (works for everyone who clones the repo).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

# =============================================================================
# USER-EDITABLE REFERENCE VALUES
# =============================================================================
# Edit these when you have measured / wind-tunnel / analytical reference
# values. They are drawn as a coloured marker on every scenario's whisker so
# you can see at a glance whether each simulated range brackets the truth.
# Set to None to hide the reference marker.

TRUE_FX = 4207.07   # [N]  reference horizontal-force value
TRUE_FY =  411.13   # [N]  reference vertical-force value

# =============================================================================

# --- Paths -------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
SIMS_ROOT  = SCRIPT_DIR / "forces_visualisation"
OUTPUT_DIR = SCRIPT_DIR / "forces_output"

# --- Folder-name patterns ----------------------------------------------------

# Parent folder, e.g.  sims_(vel_mean_14.514_aoa_mean_-14.475)
PARENT_PATTERN = re.compile(
    r"sims_\(vel_mean_(?P<vmean>-?\d+(?:\.\d+)?)"
    r"_aoa_mean_(?P<amean>-?\d+(?:\.\d+)?)\)"
)
# Case folder, e.g.  v14.391_a-14.025
CASE_PATTERN = re.compile(
    r"v(?P<v>-?\d+(?:\.\d+)?)_a(?P<a>-?\d+(?:\.\d+)?)"
)

# Tolerance when comparing floats parsed from folder names
TOL = 1e-6

# --- Plot styling ------------------------------------------------------------

COLOR_RANGE        = "#888780"   # whisker line - neutral gray
COLOR_CAP          = "#185FA5"   # end-cap ticks - blue
COLOR_MOST_LIKELY  = "#185FA5"   # most-likely marker - blue
COLOR_TRUE         = "#D85A30"   # true-value marker - coral (distinct hue)


# --- Helpers -----------------------------------------------------------------

def classify(value: float, mean: float, low: float, high: float) -> str:
    """Label a value as 'mean', 'mean - 3sd', or 'mean + 3sd'."""
    if abs(value - mean) < TOL:
        return "mean"
    if abs(value - low) < TOL:
        return "mean - 3sd"
    if abs(value - high) < TOL:
        return "mean + 3sd"
    return f"other ({value})"


def read_final_forces(path: Path) -> tuple[float, float]:
    """
    Return (fx, fy) from the last numeric row of a forces.txt file.

    The file has three header lines followed by rows of
        iteration  fx-building  fy-building
    separated by whitespace. Header lines are skipped automatically by
    failing the float() conversion.
    """
    last: tuple[float, float] | None = None
    with path.open() as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                fx = float(parts[1])
                fy = float(parts[2])
            except ValueError:
                continue  # header row
            last = (fx, fy)
    if last is None:
        raise ValueError(f"No numeric data found in {path}")
    return last


def find_sims_folders(root: Path) -> list[Path]:
    """Locate every sims_(...) folder beneath `root`, sorted by name."""
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")
    folders = [
        child for child in root.iterdir()
        if child.is_dir() and PARENT_PATTERN.match(child.name)
    ]
    return sorted(folders, key=lambda p: p.name)


def _row(quantity: str, bound: str, record: dict) -> dict:
    """Build one CSV output row from a stored case record."""
    value = record["fx"] if quantity == "fx" else record["fy"]
    return {
        "quantity":         quantity,
        "bound":            bound,
        "value [N]":        value,
        "case":             record["case"],
        "U_inlet [m/s]":    record["v"],
        "U_inlet label":    record["v_label"],
        "AoA [deg]":        record["a"],
        "AoA label":        record["a_label"],
    }


def extract_summary(sims_folder: Path):
    """
    Read every sub-case's final forces in `sims_folder` and return
        (summary_dict, csv_rows)
    or None if no usable data was found.

    summary_dict contains the numeric values needed for plotting; csv_rows
    is the 6-row (or 4-row, if no mean-mean case) list ready to write to CSV.
    """
    pm = PARENT_PATTERN.match(sims_folder.name)
    v_mean = float(pm.group("vmean"))
    a_mean = float(pm.group("amean"))

    # Collect every sub-case folder
    cases: list[tuple[float, float, Path]] = []
    for sub in sims_folder.iterdir():
        if not sub.is_dir():
            continue
        m = CASE_PATTERN.match(sub.name)
        if m:
            cases.append((float(m.group("v")), float(m.group("a")), sub))
    if not cases:
        print(f"  no sub-case folders in {sims_folder.name}, skipped")
        return None

    unique_v = sorted({c[0] for c in cases})
    unique_a = sorted({c[1] for c in cases})
    v_low, v_high = unique_v[0], unique_v[-1]
    a_low, a_high = unique_a[0], unique_a[-1]

    print(f"\nProcessing  {sims_folder.name}")
    print(f"  means:    U_inlet = {v_mean} m/s,  AoA = {a_mean} deg")
    print(f"  U range:  {v_low}  ...  {v_high}")
    print(f"  A range:  {a_low}  ...  {a_high}")
    print(f"  cases:    {len(cases)} sub-folder(s)")

    # Read final forces for every case
    records: list[dict] = []
    for v, a, sub in cases:
        ff = sub / "forces.txt"
        if not ff.exists():
            print(f"  warning: {ff.relative_to(SCRIPT_DIR)} missing - skipped")
            continue
        fx, fy = read_final_forces(ff)
        records.append({
            "case":    sub.name,
            "v":       v,
            "a":       a,
            "v_label": classify(v, v_mean, v_low, v_high),
            "a_label": classify(a, a_mean, a_low, a_high),
            "fx":      fx,
            "fy":      fy,
        })
    if not records:
        print(f"  no usable forces.txt files in {sims_folder.name}")
        return None

    fx_min_rec = min(records, key=lambda r: r["fx"])
    fx_max_rec = max(records, key=lambda r: r["fx"])
    fy_min_rec = min(records, key=lambda r: r["fy"])
    fy_max_rec = max(records, key=lambda r: r["fy"])

    most_likely = next(
        (r for r in records if r["v_label"] == "mean" and r["a_label"] == "mean"),
        None,
    )
    if most_likely is None:
        print(f"  warning: no (mean, mean) case found in {sims_folder.name} - "
              f"'most likely' rows will be omitted")

    # Build the CSV rows
    csv_rows = [_row("fx", "lowest", fx_min_rec)]
    if most_likely is not None:
        csv_rows.append(_row("fx", "most likely", most_likely))
    csv_rows.append(_row("fx", "highest", fx_max_rec))
    csv_rows.append(_row("fy", "lowest", fy_min_rec))
    if most_likely is not None:
        csv_rows.append(_row("fy", "most likely", most_likely))
    csv_rows.append(_row("fy", "highest", fy_max_rec))

    # Build the summary dict used for plotting
    summary = {
        "v_mean": v_mean,
        "a_mean": a_mean,
        "fx_min": fx_min_rec["fx"],
        "fx_max": fx_max_rec["fx"],
        "fy_min": fy_min_rec["fy"],
        "fy_max": fy_max_rec["fy"],
        "fx_mid": most_likely["fx"] if most_likely is not None else None,
        "fy_mid": most_likely["fy"] if most_likely is not None else None,
    }
    return summary, csv_rows


def write_summary_csv(rows: list[dict], v_mean: float, a_mean: float) -> Path:
    """Write a per-sims-folder summary CSV into OUTPUT_DIR."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"forces_{v_mean}_{a_mean}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> wrote {len(rows)} rows to {out.relative_to(SCRIPT_DIR)}")
    return out


# --- Plotting ----------------------------------------------------------------

def plot_component(summaries: list[dict], component: str) -> None:
    """
    Make a single forest-style plot for either 'fx' or 'fy'.
    One row per scenario, scaling automatically with len(summaries).
    """
    assert component in ("fx", "fy")
    true_val = TRUE_FX if component == "fx" else TRUE_FY
    nice_name = "Horizontal force fx" if component == "fx" else "Vertical force fy"

    n = len(summaries)
    if n == 0:
        return

    # Height grows with the number of scenarios; width is constant
    fig_h = max(2.6, 0.85 * n + 1.6)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    labels = []
    for i, s in enumerate(summaries):
        y = n - 1 - i  # first scenario on top
        lo = s[f"{component}_min"]
        hi = s[f"{component}_max"]
        mid = s[f"{component}_mid"]

        # Whisker line (lowest <-> highest)
        ax.hlines(y, lo, hi, color=COLOR_RANGE, linewidth=1.6, alpha=0.55,
                  zorder=2)
        # End caps
        cap_h = 0.18
        ax.vlines([lo, hi], y - cap_h, y + cap_h, color=COLOR_CAP,
                  linewidth=1.6, zorder=3)
        # Most-likely marker
        if mid is not None:
            ax.plot(mid, y, "o", color=COLOR_MOST_LIKELY, markersize=8,
                    zorder=4,
                    label="most likely (mean U, mean AoA)" if i == 0 else None)
        # True-value reference marker (different colour)
        if true_val is not None:
            ax.plot(true_val, y, "o", color=COLOR_TRUE, markersize=8,
                    zorder=4,
                    label="true value (reference)" if i == 0 else None)

        labels.append(f"U = {s['v_mean']} m/s\nAoA = {s['a_mean']}\u00b0")

    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(labels)))
    ax.set_xlabel(f"{component} [N]")
    title = f"{nice_name} - bounds per scenario"
    if true_val is not None:
        title += f"    (reference = {true_val:.2f} N)"
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.set_ylim(-0.7, n - 0.3)
    ax.legend(loc="best", framealpha=0.9, fontsize=9)

    fig.tight_layout()
    out = OUTPUT_DIR / f"forces_{component}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {out.relative_to(SCRIPT_DIR)}")


def plot_all(summaries: list[dict]) -> None:
    """Produce both fx and fy summary plots."""
    if not summaries:
        print("\nNothing to plot.")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nPlotting summary figures "
          f"(reference fx = {TRUE_FX} N, fy = {TRUE_FY} N)")
    plot_component(summaries, "fx")
    plot_component(summaries, "fy")


# --- Main --------------------------------------------------------------------

def main() -> None:
    sims_folders = find_sims_folders(SIMS_ROOT)
    if not sims_folders:
        raise FileNotFoundError(
            f"No 'sims_(vel_mean_..._aoa_mean_...)' folders found in {SIMS_ROOT}"
        )
    print(f"Found {len(sims_folders)} sims-folder(s) in {SIMS_ROOT.name}/")

    summaries: list[dict] = []
    for sf in sims_folders:
        result = extract_summary(sf)
        if result is None:
            continue
        summary, csv_rows = result
        write_summary_csv(csv_rows, summary["v_mean"], summary["a_mean"])
        summaries.append(summary)

    plot_all(summaries)

    print(f"\nDone. {len(summaries)} scenario(s) processed.")


if __name__ == "__main__":
    main()
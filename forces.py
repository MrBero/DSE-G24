"""
forces.py - Summarise final aerodynamic forces from CFD simulation runs.

Walks every `sims_(vel_mean_*_aoa_mean_*)` folder inside `forces_visualisation/`,
reads the last iteration of `forces.txt` for every sub-case, and writes one
summary CSV per sims-folder into `forces_output/`, named:

    forces_<meanUinlet>_<meanAOA>.csv      e.g.  forces_14.514_-14.475.csv

Each CSV has 6 rows:
    fx  lowest        (worst-case lower bound on horizontal force)
    fx  most likely   (the mean-U, mean-AoA case)
    fx  highest       (worst-case upper bound)
    fy  lowest        (same, for vertical force)
    fy  most likely
    fy  highest

The 'situation' columns record which (U_inlet, AoA) combination produced
each bound, with labels saying whether each input is mean, mean - 3 sd,
or mean + 3 sd.

Paths are resolved relative to this script's own location so the project
is portable across machines (works for everyone who clones the repo).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

# --- Paths -------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
SIMS_ROOT = SCRIPT_DIR / "forces_visualisation"
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
    """Build one output row from a stored case record."""
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


def process_sims_folder(sims_folder: Path) -> Path | None:
    """
    Process a single sims_(...) folder: read every sub-case's final forces
    and write a 6-row summary CSV named `forces_<vmean>_<amean>.csv` into
    OUTPUT_DIR. Returns the CSV path (or None if no usable data was found).
    """
    parent_match = PARENT_PATTERN.match(sims_folder.name)
    v_mean = float(parent_match.group("vmean"))
    a_mean = float(parent_match.group("amean"))

    # Collect every case folder present
    cases: list[tuple[float, float, Path]] = []
    for sub in sims_folder.iterdir():
        if not sub.is_dir():
            continue
        m = CASE_PATTERN.match(sub.name)
        if m:
            cases.append((float(m.group("v")), float(m.group("a")), sub))

    if not cases:
        print(f"  no sub-case folders found in {sims_folder.name}, skipped")
        return None

    # Smallest = mean - 3sd, largest = mean + 3sd
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
        forces_file = sub / "forces.txt"
        if not forces_file.exists():
            print(f"  warning: {forces_file.relative_to(SCRIPT_DIR)} missing - skipped")
            continue
        fx, fy = read_final_forces(forces_file)
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

    # Pick out min, max, and the (mean, mean) "most likely" case for each force
    fx_min = min(records, key=lambda r: r["fx"])
    fx_max = max(records, key=lambda r: r["fx"])
    fy_min = min(records, key=lambda r: r["fy"])
    fy_max = max(records, key=lambda r: r["fy"])

    most_likely = next(
        (r for r in records if r["v_label"] == "mean" and r["a_label"] == "mean"),
        None,
    )
    if most_likely is None:
        print(f"  warning: no (mean, mean) case found in {sims_folder.name} - "
              f"'most likely' rows will be omitted")

    summary_rows = [_row("fx", "lowest", fx_min)]
    if most_likely is not None:
        summary_rows.append(_row("fx", "most likely", most_likely))
    summary_rows.append(_row("fx", "highest", fx_max))

    summary_rows.append(_row("fy", "lowest", fy_min))
    if most_likely is not None:
        summary_rows.append(_row("fy", "most likely", most_likely))
    summary_rows.append(_row("fy", "highest", fy_max))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"forces_{v_mean}_{a_mean}.csv"
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"  -> wrote {len(summary_rows)} summary rows to "
          f"{output_path.relative_to(SCRIPT_DIR)}")
    return output_path


# --- Main --------------------------------------------------------------------

def main() -> None:
    sims_folders = find_sims_folders(SIMS_ROOT)
    if not sims_folders:
        raise FileNotFoundError(
            f"No 'sims_(vel_mean_..._aoa_mean_...)' folders found in {SIMS_ROOT}"
        )

    print(f"Found {len(sims_folders)} sims-folder(s) in {SIMS_ROOT.name}/")

    written = []
    for sf in sims_folders:
        result = process_sims_folder(sf)
        if result is not None:
            written.append(result)

    print(f"\nDone. {len(written)} CSV file(s) written to "
          f"{OUTPUT_DIR.relative_to(SCRIPT_DIR)}/")


if __name__ == "__main__":
    main()
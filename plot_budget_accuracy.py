"""
plot_budget_accuracy.py
=======================

Single-figure plot for the "how many points do we need" study: one initial step
(1 point) plus one adaptive step of N points, swept over N. Plots the final
drag-force F_y relative error against the momentum integral on the CFD truth, as
mean +- 1 std over seeds, in the same band style as the other sensitivity plots.

The numbers below are the final-phase F_y-vs-momentum results (mean, std) per
total point count, taken straight from the study output. Edit DATA to update.
"""

import numpy as np
import matplotlib.pyplot as plt

# total points -> (mean F_y error [%], std [%]) vs the momentum integral on truth
DATA = {
    80:  (45.52, 13.68),
    160: (23.36, 18.16),
    240: (5.40, 10.79),
    320: (4.79, 9.92),
    400: (4.71, 7.91),
    480: (-1.95, 7.39),
    560: (-3.42, 6.70),
}

BAND = 0.05          # +-5% acceptance band
ADOPTED = 400        # highlight the adopted budget
OUT = "plots_400/Fy_vs_points_vs_momentum.png"


def main():
    pts = np.array(sorted(DATA))
    mean = np.array([DATA[p][0] for p in pts])
    std = np.array([DATA[p][1] for p in pts])

    fig, ax = plt.subplots(figsize=(8, 5))

    # mean line + 1 std band
    ax.plot(pts, mean, "-o", color="tab:blue", markersize=5, label="mean over seeds")
    ax.fill_between(pts, mean - std, mean + std, color="tab:blue", alpha=0.18,
                    label=r"$\pm1$ std")

    # exact line and +-5% acceptance band
    ax.axhline(0.0, ls="--", color="0.4", lw=1.0)
    ax.axhspan(-100 * BAND, 100 * BAND, color="0.5", alpha=0.12, label=r"$\pm5\%$ band")

    # mark the adopted budget
    if ADOPTED in DATA:
        ax.axvline(ADOPTED, ls=":", color="0.5", lw=1.2)
        ax.annotate(f"{ADOPTED} pts", (ADOPTED, ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(4, -12),
                    fontsize=8, color="0.6")

    ax.set_xlabel("total sampling points")
    ax.set_ylabel(r"$F_y$ relative error vs momentum integral [\%]")
    ax.set_title(r"$F_y$ error against the momentum integral on the CFD truth "
                 "vs total points")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    fig.tight_layout()
    import os
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT}")
    plt.show()


if __name__ == "__main__":
    main()

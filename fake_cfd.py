"""
generate_fake_cfd.py

Creates a folder of fake CFD CSV files for testing the ETKF pipeline.
Each file has the same x, y, u, v format we expect from real Fluent exports.

The "physics" here is fake — flow is a uniform stream tilted by AoA, with
a smooth spatial perturbation. Good enough to make the ETKF work,
not meant to be realistic.

Run once to populate a data folder, then point main.py at it.
"""

import os
import numpy as np


def make_fake_flow(x, y, U_inlet, alpha_deg):
    """
    Fake flow field: uniform stream tilted by alpha, with a small
    spatial perturbation. Returns u, v arrays of length len(x).
    """
    alpha_rad = np.deg2rad(alpha_deg)
    
    # spatial perturbation that's smooth and bounded in [-1, 1]
    perturbation = 0.05 * np.sin(0.3 * x) * np.cos(0.5 * y)
    
    u = U_inlet * np.cos(alpha_rad) * (1.0 + perturbation)
    v = U_inlet * np.sin(alpha_rad) * (1.0 + perturbation)
    
    return u, v


def write_cfd_file(path, x, y, u, v):
    """Write a CSV file with header line 'x,y,u,v' and one row per cell."""
    data = np.column_stack([x, y, u, v])
    with open(path, "w") as f:
        f.write("x,y,u,v\n")
        np.savetxt(f, data, delimiter=",", fmt="%.6f")


if __name__ == "__main__":
    # mesh: 30 x 10 grid of cells
    nx, ny = 30, 10
    x_grid, y_grid = np.meshgrid(
        np.linspace(0, 30, nx),
        np.linspace(0, 10, ny),
    )
    x = x_grid.flatten()
    y = y_grid.flatten()
    print(f"Mesh has {len(x)} cells")
    
    # ensemble: 12 members spanning the parameter ranges
    rng = np.random.default_rng(0)
    N = 100
    U_inlet_lst = rng.uniform(3.5, 7.5, N)
    alpha_lst   = rng.uniform(-5.0, 5.0, N)
    
    # truth case: parameters not equal to any member
    truth_U_inlet = 5.7
    truth_alpha   = 2.3
    
    # output folder
    folder = "fakedata"
    os.makedirs(folder, exist_ok=True)
    
    # write ensemble files
    path_lst = []
    for i in range(N):
        u, v = make_fake_flow(x, y, U_inlet_lst[i], alpha_lst[i])
        path = os.path.join(folder, f"member_{i:02d}.csv")
        write_cfd_file(path, x, y, u, v)
        path_lst.append(path)
        print(f"  wrote {path}: U={U_inlet_lst[i]:.3f}, alpha={alpha_lst[i]:.3f}")
    
    # write truth file
    u_truth, v_truth = make_fake_flow(x, y, truth_U_inlet, truth_alpha)
    truth_path = os.path.join(folder, "truth.csv")
    write_cfd_file(truth_path, x, y, u_truth, v_truth)
    print(f"  wrote {truth_path}: U={truth_U_inlet}, alpha={truth_alpha}")
    

    print(f"U_inlet: {U_inlet_lst} \n alpha: {alpha_lst} \n truth params: {truth_U_inlet, truth_alpha}")
    print("Done. Run main.py next.")
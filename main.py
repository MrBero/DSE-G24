import os
import json
import numpy as np
import matplotlib.pyplot as plt

from load import load_all_cfds, load_one_cfd
from drone_sampling import make_drone_observations
from etkf import etkf
from plotter import plot_just_field, plot_error_field, plot_hist
from Seppe_drone import pick_informative_drones

from alive_progress import alive_bar


# parameters used by generate_fake_cfd.py
case_folder = 'sim'
NOISE_STD = 0.15
fake = False

def main(X_positions, X_ensemble, U_inlet_lst, alpha_lst, truth_U_inlet, truth_alpha, truth_path, point_distribution, dims, n_samples):
    #point_distribution = 'optimized'

    if point_distribution == 'random':
        seed = sum([ord(char) for char in 'PEACH_VIBE'])
        drone_xy = sample_trapezium(75, 20, 50, 90 - 14.5, n_samples, excl_w = 15, excl_h = 20) + np.array([107.5,0])
        # print(drone_xy)
        # np.random.seed(seed)
        # drone_xy = np.vstack([np.random.random(n_samples) * dims[0], 
        #                     np.random.random(n_samples) * dims[1] - dims[1]/2]).T
    elif point_distribution == 'optimized':
        n_cells = (X_ensemble.shape[0] - 2) // 2
        xcoord = X_positions[0:2 * n_cells:2, 0]
        ycoord = X_positions[0:2 * n_cells:2, 1]
        drone_xy, score = pick_informative_drones(X_ensemble, xcoord, ycoord, NOISE_STD**2, n_samples, U_inf_row=-2, alpha_row=-1)

        plt.figure(figsize=(10, 8))
        # Use 'scatter' with your sorted arrays
        # 's' controls point size, 'c' is the color depth, 'cmap' is the color theme
        plt.scatter(xcoord, ycoord, c=score, s=10, cmap='cool', edgecolors='none')
        plt.colorbar(label='Intensity Score')
        plt.title("Point-based Heat Map")
        plt.show()
    else:
        with open(os.path.join('sample-distributions', point_distribution), 'r') as f:
            drone_xy = np.array(json.load(f)['xy'])
            # print(drone_xy['xy'])
    print('Points sampled!')
    # generate synthetic measurements from the truth case
    y_measured, obs_indices, drone_xy_snapped = make_drone_observations(
        truth_path, drone_xy, noise_std=NOISE_STD, seed=42
    )
    
    #calculate the mean of all CFDs before
    Xinitial_mean = np.mean(X_ensemble, axis=1)
    Xinitial_mean_even = Xinitial_mean[::2]
    Xinitial_mean_odd = Xinitial_mean[1::2]
    # X_mean_reshaped = np.vstack([X_mean_even, X_mean_odd])
    Xinitial_mean_mags = np.sqrt(Xinitial_mean_even**2 + Xinitial_mean_odd**2)
    
    # slice predicted observations from the ensemble
    y_pred = X_ensemble[obs_indices, :]
    
    # observation noise covariance (assume same noise level for all sensors)
    R = np.full(len(y_measured), NOISE_STD ** 2)
    
    print("Running ETKF...")
    # run ETKF
    X_analysis, dX_analysis, mean_corr = etkf(X_ensemble, y_measured, y_pred, R)
    # extract assimilated parameters
    analysis_inlet = X_analysis[-2, :]
    analysis_alpha = X_analysis[-1, :]


    #calculate the mean of all CFDs after
    X_mean = np.mean(X_analysis, axis=1)
    X_mean_even = X_mean[::2]
    X_mean_odd = X_mean[1::2]
    # X_mean_reshaped = np.vstack([X_mean_even, X_mean_odd])
    X_mean_mags = np.sqrt(X_mean_even**2 + X_mean_odd**2)

    
    print("=== Results ===")
    print(f'POINT DISTRIBUTION: {point_distribution}')
    print(f"Number of samples: {len(y_measured)/2}")
    print(f"Size of state matrix: {X_analysis.shape}\n")

    print(f"Truth inlet velocity: {truth_U_inlet:.3f} [m/s]")
    print(f"Truth AoA: {truth_alpha:.3f} [deg]\n")

    print(f"Initial mean inlet: {U_inlet_lst.mean():.3f} [m/s]")
    print(f"Initial mean AoA: {alpha_lst.mean():.3f} [deg]\n")
    
    print(f"Final mean inlet: {analysis_inlet.mean():.3f} [m/s]")
    print(f"Final std inlet: {analysis_inlet.std():.3f} [m/s]\n")
    
    print(f"Final mean AoA: {analysis_alpha.mean():.3f} [deg]")
    print(f"Final std AoA: {analysis_alpha.std():.3f} [deg]\n")

    # print(dX_analysis)
    # print(mean_corr)
    # success check
    inlet_improvement = abs(analysis_inlet.mean() - truth_U_inlet) < abs(U_inlet_lst.mean() - truth_U_inlet)
    alpha_improvement = abs(analysis_alpha.mean() - truth_alpha)   < abs(alpha_lst.mean()   - truth_alpha)
    
    print(f"Inlet improved: {inlet_improvement}")
    print(f"AoA improved: {alpha_improvement}")

    xtrue, ytrue, utrue, vtrue = load_one_cfd(truth_path)
    Xtrue_positions = np.vstack([xtrue, ytrue]).T
    Xtrue_mags = np.sqrt(utrue**2 + vtrue**2)
    # plot_hist(U_inlet_lst, analysis_inlet, truth_U_inlet, alpha_lst, analysis_alpha, truth_alpha)

    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # fig.tight_layout()
    # plot_just_field(axes[0], Xtrue_positions, Xtrue_mags, title = 'Truth', flat = False)
    # plot_just_field(axes[1], X_positions, Xinitial_mean_mags, title = "Initial Means")
    # plot_just_field(axes[2], X_positions, X_mean_mags, title = "Final Means")
    # plot_error_field(X_positions, Xinitial_mean_mags, X_mean_mags, drone_xy)
    plt.show()
    return analysis_inlet.mean(), analysis_inlet.std(), analysis_alpha.mean(), analysis_alpha.std()
    
#load cfds
def load_main():
    U_inlet_lst = []
    alpha_lst = []
    for f in os.listdir(directory):
        # print(f)
        string = f.split(sep='_')
        # print(string)
        U_inlet_lst.append(float(string[1]))
        alpha_lst.append(float(string[3]))
    U_inlet_lst = np.array(U_inlet_lst)
    alpha_lst = np.array(alpha_lst)

    path_lst = [os.path.join(directory, file, 'solution_data.csv') for file in os.listdir(directory)]
    #truths 
    truth_U_inlet, truth_alpha = 14.5, -14.5

    X_positions, X_ensemble, _, _ = load_all_cfds(path_lst, U_inlet_lst, alpha_lst)
    print('CFDs loaded!')
    return X_positions, X_ensemble, U_inlet_lst, alpha_lst, truth_U_inlet, truth_alpha

import numpy as np

def trapezium_verts(h, b1, b2, angle_deg):
    """Returns 4 vertices of a symmetric trapezoid rotated around center."""
    verts = np.array([
        [-b1/2,  h/2],   # bottom-left
        [ b1/2,  h/2],   # bottom-right
        [ b2/2, -h/2],   # top-right
        [-b2/2, -h/2],   # top-left
    ])
    ang = np.radians(angle_deg)
    R = np.array([[np.cos(ang), -np.sin(ang)],
                  [np.sin(ang),  np.cos(ang)]])
    return verts @ R.T

def sample_trapezium(h, b1, b2, angle_deg, n, excl_w=0, excl_h=0):
    """
    excl_w, excl_h: width and height of axis-aligned exclusion rectangle at origin.
    """
    verts = trapezium_verts(h, b1, b2, angle_deg)
    A, B, C, D = verts

    area1 = 0.5 * abs(np.cross(B - A, C - A))
    area2 = 0.5 * abs(np.cross(C - A, D - A))
    total = area1 + area2

    # account for exclusion zone reducing the valid area
    excl_area = excl_w * excl_h
    fill_ratio = 1 - excl_area / total  # expected fraction that passes
    oversample = max(int(1 / fill_ratio * 1.3), 2) if fill_ratio < 1 else 1

    def sample_triangle(P, Q, R, k):
        r1 = np.random.random((k, 1))
        r2 = np.random.random((k, 1))
        mask = (r1 + r2) > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        return P + r1 * (Q - P) + r2 * (R - P)

    hw, hh = excl_w / 2, excl_h / 2
    out = []

    while len(out) < n:
        k = (n - len(out)) * oversample
        n1 = int(k * area1 / total)
        n2 = k - n1
        pts = np.vstack([
            sample_triangle(A, B, C, n1),
            sample_triangle(A, C, D, n2),
        ])
        # exclude axis-aligned rectangle at center
        in_excl = (
            (pts[:, 0] >= -hw) & (pts[:, 0] <= hw) &
            (pts[:, 1] >= -hh) & (pts[:, 1] <= hh)
        )
        out.append(pts[~in_excl])

    return np.vstack(out)[:n]

if __name__ == "__main__":
    directory = os.path.join(case_folder, 'CORRECTED_simulation_outputs')
    truth_path = os.path.join(case_folder, 'solution_data_truth14.5.csv')

    loaded = load_main()
    results = []
    with alive_bar(100) as bar:
        for n in np.arange(50, 101, 1):
            inlet_mean, inlet_std, alpha_mean, alpha_std = main(*loaded,
                                                                truth_path = truth_path,
                                                                point_distribution = 'optimized',
                                                                dims = (800, 500),
                                                                n_samples = n)
            results.append([n, inlet_mean, inlet_std, alpha_mean, alpha_std])
            bar()
    results = np.array(results)
    plt.plot(results[:,0], results[:, 2])
    plt.show()
    

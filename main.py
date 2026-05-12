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

def main(directory, truth_path, point_distribution, dims, n_samples):
    #simulations
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

    #point_distribution = 'optimized'
    if point_distribution == 'random':
        seed = sum([ord(char) for char in 'PEACH_VIBE'])
        np.random.seed(seed)
        drone_xy = np.vstack([np.random.random(n_samples) * dims[0], 
                            np.random.random(n_samples) * dims[1] - dims[1]/2]).T
    elif point_distribution == 'optimized':
        xcoord = X_positions[:, 0]
        ycoord = X_positions[:, 1]
        drone_xy, = pick_informative_drones(X_ensemble, xcoord, ycoord, NOISE_STD, U_inf_row=-2, alpha_row=-1)
        
    else:
        with open(os.path.join('sample-distributions', point_distribution), 'r') as f:
            drone_xy = np.array(json.load(f)['xy'])
            # print(drone_xy['xy'])

    # generate synthetic measurements from the truth case
    y_measured, obs_indices, drone_xy_snapped = make_drone_observations(
        truth_path, drone_xy, noise_std=NOISE_STD, seed=42
    )
    
    #calculate the mean of all CFDs before
    Xinitial_mean = np.mean(X_ensemble, axis=1)
    Xinitial_mean_even = Xinitial_mean[::2]
    Xinitial_mean_odd = Xinitial_mean[1::2]
    # X_mean_reshaped = np.vstack([X_mean_even, X_mean_odd])
    Xinitial_mean_mags = np.sqrt(np.pow(Xinitial_mean_even,2) + np.pow(Xinitial_mean_odd, 2))
    
    # slice predicted observations from the ensemble
    y_pred = X_ensemble[obs_indices, :]
    
    # observation noise covariance (assume same noise level for all sensors)
    R = np.full(len(y_measured), NOISE_STD ** 2)
    
    with alive_bar() as bar:
        print("Running ETKF...")
        # run ETKF
        X_analysis, dX_analysis, mean_corr = etkf(X_ensemble, y_measured, y_pred, R, bar)
    # extract assimilated parameters
    analysis_inlet = X_analysis[-2, :]
    analysis_alpha = X_analysis[-1, :]


    #calculate the mean of all CFDs after
    X_mean = np.mean(X_analysis, axis=1)
    X_mean_even = X_mean[::2]
    X_mean_odd = X_mean[1::2]
    # X_mean_reshaped = np.vstack([X_mean_even, X_mean_odd])
    X_mean_mags = np.sqrt(np.pow(X_mean_even,2) + np.pow(X_mean_odd, 2))

    
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
    plot_hist(U_inlet_lst, analysis_inlet, truth_U_inlet, alpha_lst, analysis_alpha, truth_alpha)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.tight_layout()
    plot_just_field(axes[0], Xtrue_positions, Xtrue_mags, title = 'Truth', flat = False)
    plot_just_field(axes[1], X_positions, Xinitial_mean_mags, title = "Initial Means")
    plot_just_field(axes[2], X_positions, X_mean_mags, title = "Final Means")
    plot_error_field(X_positions, Xinitial_mean_mags, X_mean_mags, drone_xy_snapped)
    
    plt.show()
    


if __name__ == "__main__":
    directory = os.path.join(case_folder, 'CORRECTED_simulation_outputs')
    truth_path = os.path.join(case_folder, 'solution_data_truth14.5.csv')
    
    main(directory = directory, 
         truth_path = truth_path,
         point_distribution = 'random',
         dims = (800, 500),
         n_samples = 40)
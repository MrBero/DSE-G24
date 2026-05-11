import os
import json
import numpy as np
import matplotlib.pyplot as plt

from load import load_all_cfds, load_one_cfd
from drone_sampling import make_drone_observations
from etkf import etkf
from plotter import plot_just_field, plot_error_field, plot_hist

from alive_progress import alive_bar


# parameters used by generate_fake_cfd.py
case_folder = 'sim'
NOISE_STD = 0.15
fake = False

def main():
    U_inlet_lst = []
    alpha_lst = []
    directory = os.path.join(case_folder, 'simulation_outputs')
    for f in os.listdir(directory):
        # print(f)
        string = f.split(sep='_')
        # print(string)
        U_inlet_lst.append(float(string[1]))
        alpha_lst.append(float(string[3]))
    U_inlet_lst = np.array(U_inlet_lst)
    alpha_lst = np.array(alpha_lst)
    path_lst = [os.path.join(case_folder, 'simulation_outputs', file, 'solution_data.csv') for file in os.listdir(directory)]
    truth_path = os.path.join(case_folder, 'solution_data_truth13.csv')
    # print(path_lst, truth_path, U_inlet_lst, alpha_lst)
    X_positions, X_ensemble = load_all_cfds(path_lst, U_inlet_lst, alpha_lst)
    truth_U_inlet, truth_alpha = 13, -13
    print('CFDs loaded!')

    #drone distributions
    dims = (800, 500)
    point_distribution = 'ENKF_points.json'
    # n_samples = 20
    # drone_xy = np.vstack([np.random.random(n_samples) * dims[0], 
    #                       np.random.random(n_samples) * dims[1] - dims[1]/2]).T
    with open(os.path.join('sample-distributions', point_distribution), 'r') as f:
        drone_xy = np.array(json.load(f)['xy'])
        # print(drone_xy['xy'])

    # generate synthetic measurements from the truth case
    y_measured, obs_indices, drone_xy_snapped = make_drone_observations(
        truth_path, drone_xy, noise_std=NOISE_STD, seed=42
    )
    
    #calculate the mean of all CFDs before
    Xtrue_mean = np.mean(X_ensemble, axis=1)
    Xtrue_mean_even = Xtrue_mean[::2]
    Xtrue_mean_odd = Xtrue_mean[1::2]
    # X_mean_reshaped = np.vstack([X_mean_even, X_mean_odd])
    Xtrue_mean_mags = np.sqrt(np.pow(Xtrue_mean_even,2) + np.pow(Xtrue_mean_odd, 2))
    
    
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
    print(f"Number of samples: {len(y_measured)}")
    print(f"Number of elements per CFD: {X_analysis.shape}")
    print(f"Truth inlet velocity: {truth_U_inlet:.3f}")
    print(f"Initial mean inlet: {U_inlet_lst.mean():.3f}")
    print(f"Final mean inlet: {analysis_inlet.mean():.3f}")
    print(f"Final std inlet: {analysis_inlet.std():.3f}")
    print(f"Truth AoA: {truth_alpha:.3f}")
    print(f"Forecast mean AoA: {alpha_lst.mean():.3f}")
    print(f"Final mean AoA: {analysis_alpha.mean():.3f}")
    print(f"Final std AoA: {analysis_alpha.std():.3f}")

    # print(dX_analysis)
    # print(mean_corr)
    
    # success check
    inlet_improvement = abs(analysis_inlet.mean() - truth_U_inlet) < abs(U_inlet_lst.mean() - truth_U_inlet)
    alpha_improvement = abs(analysis_alpha.mean() - truth_alpha)   < abs(alpha_lst.mean()   - truth_alpha)
    
    print(f"Inlet improved: {inlet_improvement}")
    print(f"AoA improved: {alpha_improvement}")

    plot_hist(U_inlet_lst, analysis_inlet, truth_U_inlet, alpha_lst, analysis_alpha, truth_alpha)
    # plot_just_field(X_positions, X_mean_mags, title = "Initial Means")
    # plot_just_field(X_positions, X_mean_mags, title = "Final Means")
    plot_error_field(X_positions, Xtrue_mean_mags, X_mean_mags, drone_xy_snapped)
    
    plt.show()
    


if __name__ == "__main__":
    main()
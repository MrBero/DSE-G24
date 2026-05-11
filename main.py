"""
main.py

End-to-end test of the ETKF pipeline using fake CFD data.

Run generate_fake_cfd.py first to populate the data folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from load import load_all_cfds, load_one_cfd
from drone_sampling import make_drone_observations
from etkf import etkf
from plotter import plot_field, plot_hist

from alive_progress import alive_bar


# parameters used by generate_fake_cfd.py
case_folder = 'solutions'
NOISE_STD = 0.15
fake = True

def main():
    if fake: 
        N = 12
        # paths and parameters
        path_lst = [os.path.join("fakedata", f"member_{i:02d}.csv") for i in range(N)]
        truth_path = os.path.join("fakedata", "truth.csv")
        
        U_inlet_lst = np.load(os.path.join("fakedata", "U_inlet_lst.npy"))
        alpha_lst   = np.load(os.path.join("fakedata", "alpha_lst.npy"))
        truth_params = np.load(os.path.join("fakedata", "truth_params.npy"))
        truth_U_inlet, truth_alpha = truth_params[0], truth_params[1]
        # load ensemble
        X_positions, X_ensemble = load_all_cfds(path_lst, U_inlet_lst, alpha_lst)
        print(f"X_ensemble shape: {X_ensemble.shape}")
        print(f"Forecast inlet velocities: {U_inlet_lst}")
        print(f"Forecast mean inlet:       {U_inlet_lst.mean():.3f}")
        print(f"Forecast AoA:              {alpha_lst}")
        print(f"Forecast mean AoA:         {alpha_lst.mean():.3f}\n")

    else: 
       path_lst = [os.path.join(case_folder, file) for file in os.listdir(case_folder)]
       truth_path = os.path.join("fakedata", "truth.csv")
       U_inlet_lst = []
       alpha_lst = []
       X_positions, X_ensemble = load_all_cfds(path_lst, U_inlet_lst, alpha_lst)
    print(path_lst)

    # hand-pick drone locations inside the domain (0-30 in x, 0-10 in y, change later)
    dims = (30, 10)
    n_samples = 100
    drone_xy = np.vstack([np.random.random(n_samples) * dims[0], 
                          np.random.random(n_samples) * dims[1]]).T
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
    plot_field(X_positions, Xtrue_mean_mags, drone_xy_snapped)
    
    
    # slice predicted observations from the ensemble
    y_pred = X_ensemble[obs_indices, :]
    
    # observation noise covariance (assume same noise level for all sensors)
    R = np.full(len(y_measured), NOISE_STD ** 2)
    
    with alive_bar() as bar:
        print("Running ETKF...")
        # run ETKF
        X_analysis = etkf(X_ensemble, y_measured, y_pred, R, bar)
    # extract assimilated parameters
    analysis_inlet = X_analysis[-2, :]
    analysis_alpha = X_analysis[-1, :]


    #calculate the mean of all CFDs after
    X_mean = np.mean(X_analysis, axis=1)
    X_mean_even = X_mean[::2]
    X_mean_odd = X_mean[1::2]
    # X_mean_reshaped = np.vstack([X_mean_even, X_mean_odd])
    X_mean_mags = np.sqrt(np.pow(X_mean_even,2) + np.pow(X_mean_odd, 2))
    print(X_mean_mags.shape)

    
    print("=== Results ===")
    print(f"Number of measurements: {len(y_measured)}")
    print(f"Number of elements per CFD: {X_analysis.shape}")
    print(f"Truth inlet velocity: {truth_U_inlet:.3f}")
    print(f"Forecast mean inlet: {U_inlet_lst.mean():.3f}")
    print(f"Analysis mean inlet: {analysis_inlet.mean():.3f}")
    print(f"Analysis std inlet: {analysis_inlet.std():.3f}")
    print(f"Truth AoA: {truth_alpha:.3f}")
    print(f"Forecast mean AoA: {alpha_lst.mean():.3f}")
    print(f"Analysis mean AoA: {analysis_alpha.mean():.3f}")
    print(f"Analysis std AoA: {analysis_alpha.std():.3f}")
    
    # success check
    inlet_improvement = abs(analysis_inlet.mean() - truth_U_inlet) < abs(U_inlet_lst.mean() - truth_U_inlet)
    alpha_improvement = abs(analysis_alpha.mean() - truth_alpha)   < abs(alpha_lst.mean()   - truth_alpha)
    
    print(f"Inlet improved: {inlet_improvement}")
    print(f"AoA improved: {alpha_improvement}")

    plot_hist(U_inlet_lst, analysis_inlet, truth_U_inlet, alpha_lst, analysis_alpha, truth_alpha)
    
    plt.show()
    


if __name__ == "__main__":
    main()
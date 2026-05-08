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


# parameters used by generate_fake_cfd.py
FOLDER = "fakedata"
N = 12
NOISE_STD = 0.15


def main():
    # paths and parameters
    path_lst = [os.path.join(FOLDER, f"member_{i:02d}.csv") for i in range(N)]
    truth_path = os.path.join(FOLDER, "truth.csv")
    
    U_inlet_lst = np.load(os.path.join(FOLDER, "U_inlet_lst.npy"))
    alpha_lst   = np.load(os.path.join(FOLDER, "alpha_lst.npy"))
    truth_params = np.load(os.path.join(FOLDER, "truth_params.npy"))
    truth_U_inlet, truth_alpha = truth_params[0], truth_params[1]
    
    # load ensemble
    X_positions, X_ensemble = load_all_cfds(path_lst, U_inlet_lst, alpha_lst)
    print(f"X_ensemble shape: {X_ensemble.shape}")
    print(f"Forecast inlet velocities: {U_inlet_lst}")
    print(f"Forecast mean inlet:       {U_inlet_lst.mean():.3f}")
    print(f"Forecast AoA:              {alpha_lst}")
    print(f"Forecast mean AoA:         {alpha_lst.mean():.3f}\n")
    

    # hand-pick drone locations inside the domain (0-30 in x, 0-10 in y, change later)
    dims = (30, 10)
    n_samples = 100
    drone_xy = np.vstack([np.random.random(n_samples) * dims[0], 
                          np.random.random(n_samples) * dims[1]]).T
    
    # generate synthetic measurements from the truth case
    y_measured, obs_indices, drone_xy_snapped = make_drone_observations(
        truth_path, drone_xy, noise_std=NOISE_STD, seed=42
    )
    
    # slice predicted observations from the ensemble
    y_pred = X_ensemble[obs_indices, :]
    
    # observation noise covariance (assume same noise level for all sensors)
    R = np.full(len(y_measured), NOISE_STD ** 2)
    
    # run ETKF
    X_analysis = etkf(X_ensemble, y_measured, y_pred, R)
    # extract assimilated parameters
    analysis_inlet = X_analysis[-2, :]
    analysis_alpha = X_analysis[-1, :]

    #calculate the mean of all CFDs before
    Xtrue_mean = np.mean(X_ensemble, axis=1)
    Xtrue_mean_even = Xtrue_mean[::2]
    Xtrue_mean_odd = Xtrue_mean[1::2]
    # X_mean_reshaped = np.vstack([X_mean_even, X_mean_odd])
    Xtrue_mean_mags = np.sqrt(np.pow(Xtrue_mean_even,2) + np.pow(Xtrue_mean_odd, 2))

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
    
    # # quick visual: histograms before vs after
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = 16
    axes[0].hist(U_inlet_lst, bins=bins, alpha=0.5, label='forecast', color='blue')
    axes[0].hist(analysis_inlet, bins=bins, alpha=0.5, label='analysis', color='red')
    axes[0].axvline(truth_U_inlet, color='black', linestyle='--', label='truth')
    axes[0].set_xlabel('Inlet velocity')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].set_title('Inlet velocity: before vs after')
    
    axes[1].hist(alpha_lst, bins=bins, alpha=0.5, label='forecast', color='blue')
    axes[1].hist(analysis_alpha, bins=bins, alpha=0.5, label='analysis', color='red')
    axes[1].axvline(truth_alpha, color='black', linestyle='--', label='truth')
    axes[1].set_xlabel('AoA')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].set_title('AoA: before vs after')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "assimilation_results.png"), dpi=100)


    # print(X_analysis.shape)
    # print(np.min(Xtrue_mean_mags - X_mean_mags))
    # fig, ax = plt.subplots()
    # fig.suptitle("X_true to X after ENKF")
    # sc = ax.scatter(X_positions[::2,0], X_positions[::2,1], c=Xtrue_mean_mags - X_mean_mags, cmap='RdBu',
    #                 vmin=np.min(Xtrue_mean_mags[:-2:] - X_mean_mags[:-2:]), vmax=0)
    # ax.scatter(drone_xy_snapped[:,0], drone_xy_snapped[:,1], c='green')
    
    # plt.colorbar(sc, ax=ax)
    
    # fig2, ax2 = plt.subplots()
    # ax2.scatter(X_positions[::2,0], X_positions[::2,1],  c=Xtrue_mean_mags, cmap='RdBu')
    plt.show()


if __name__ == "__main__":
    main()
"""
drone_count_study.py

Experiment: how does the assimilation accuracy depend on the number of drones?

For each drone count, run multiple trials with different random drone placements,
record the error in the assimilated parameters, and plot the trend.

Run generate_fake_cfd.py first to populate the fakedata folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from load import load_all_cfds, load_one_cfd
from drone_sampling import make_drone_observations
from etkf import etkf


FOLDER = "fakedata"
N = 12
NOISE_STD = 0.15

# experiment settings
DRONE_COUNTS = [1, 2, 4, 8, 16, 32, 64]
N_TRIALS = 10
DOMAIN_X = (0, 30)
DOMAIN_Y = (0, 10)


def random_drones(n_drones, seed):
    """Generate n_drones random (x, y) positions inside the domain."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(DOMAIN_X[0], DOMAIN_X[1], n_drones)
    y = rng.uniform(DOMAIN_Y[0], DOMAIN_Y[1], n_drones)
    return np.column_stack([x, y])


def run_single_trial(X_ensemble, truth_path, n_drones, trial_seed):
    """Run one assimilation with n_drones drones placed randomly."""
    drone_xy = random_drones(n_drones, seed=trial_seed)
    
    y_measured, obs_indices, _ = make_drone_observations(
        truth_path, drone_xy, noise_std=NOISE_STD,
        seed=trial_seed + 1000  # different seed for noise
    )
    
    y_pred = X_ensemble[obs_indices, :]
    R = np.full(len(y_measured), NOISE_STD ** 2)
    
    X_analysis = etkf(X_ensemble, y_measured, y_pred, R)
    
    inlet_mean = X_analysis[-2, :].mean()
    inlet_std  = X_analysis[-2, :].std()
    alpha_mean = X_analysis[-1, :].mean()
    alpha_std  = X_analysis[-1, :].std()
    
    return inlet_mean, inlet_std, alpha_mean, alpha_std


def main():
    # load ensemble once
    path_lst = [os.path.join(FOLDER, f"member_{i:02d}.csv") for i in range(N)]
    truth_path = os.path.join(FOLDER, "truth.csv")
    
    U_inlet_lst = np.load(os.path.join(FOLDER, "U_inlet_lst.npy"))
    alpha_lst   = np.load(os.path.join(FOLDER, "alpha_lst.npy"))
    truth_params = np.load(os.path.join(FOLDER, "truth_params.npy"))
    truth_U_inlet, truth_alpha = truth_params[0], truth_params[1]
    
    _, X_ensemble = load_all_cfds(path_lst, U_inlet_lst, alpha_lst)
    
    print(f"Ensemble loaded: {X_ensemble.shape}")
    print(f"Truth: U_inlet = {truth_U_inlet}, AoA = {truth_alpha}")
    print()
    
    # storage for results
    inlet_errors = np.zeros((len(DRONE_COUNTS), N_TRIALS))
    alpha_errors = np.zeros((len(DRONE_COUNTS), N_TRIALS))
    inlet_spreads = np.zeros((len(DRONE_COUNTS), N_TRIALS))
    alpha_spreads = np.zeros((len(DRONE_COUNTS), N_TRIALS))
    
    # main loop
    for i, n_drones in enumerate(DRONE_COUNTS):
        for trial in range(N_TRIALS):
            inlet_mean, inlet_std, alpha_mean, alpha_std = run_single_trial(
                X_ensemble, truth_path, n_drones, trial_seed=trial
            )
            inlet_errors[i, trial]  = abs(inlet_mean - truth_U_inlet)
            alpha_errors[i, trial]  = abs(alpha_mean - truth_alpha)
            inlet_spreads[i, trial] = inlet_std
            alpha_spreads[i, trial] = alpha_std
        
        print(f"  n_drones={n_drones:3d}: "
              f"inlet_err={inlet_errors[i].mean():.4f}±{inlet_errors[i].std():.4f}, "
              f"alpha_err={alpha_errors[i].mean():.4f}±{alpha_errors[i].std():.4f}")
    
    # compute means and stds across trials
    inlet_err_mean = inlet_errors.mean(axis=1)
    inlet_err_std  = inlet_errors.std(axis=1)
    alpha_err_mean = alpha_errors.mean(axis=1)
    alpha_err_std  = alpha_errors.std(axis=1)
    inlet_spread_mean = inlet_spreads.mean(axis=1)
    alpha_spread_mean = alpha_spreads.mean(axis=1)
    
    # plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # top row: error
    ax = axes[0, 0]
    ax.errorbar(DRONE_COUNTS, inlet_err_mean, yerr=inlet_err_std,
                marker='o', capsize=4, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('Number of drones')
    ax.set_ylabel('|Assimilated - truth| (m/s)')
    ax.set_title('Inlet velocity: error vs drone count')
    ax.grid(True, which='both', alpha=0.3)
    
    ax = axes[0, 1]
    ax.errorbar(DRONE_COUNTS, alpha_err_mean, yerr=alpha_err_std,
                marker='o', capsize=4, color='red')
    ax.set_xscale('log')
    ax.set_xlabel('Number of drones')
    ax.set_ylabel('|Assimilated - truth| (deg)')
    ax.set_title('AoA: error vs drone count')
    ax.grid(True, which='both', alpha=0.3)
    
    # bottom row: ensemble spread (uncertainty after assimilation)
    ax = axes[1, 0]
    ax.plot(DRONE_COUNTS, inlet_spread_mean, marker='o', color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('Number of drones')
    ax.set_ylabel('Std of analysis ensemble (m/s)')
    ax.set_title('Inlet velocity: spread vs drone count')
    ax.grid(True, which='both', alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(DRONE_COUNTS, alpha_spread_mean, marker='o', color='red')
    ax.set_xscale('log')
    ax.set_xlabel('Number of drones')
    ax.set_ylabel('Std of analysis ensemble (deg)')
    ax.set_title('AoA: spread vs drone count')
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "drone_count_study.png"), dpi=100)
    plt.show()


if __name__ == "__main__":
    main()
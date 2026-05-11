import os
import matplotlib.pyplot as plt
import numpy as np

def plot_just_field(X_positions, Xtrue_mean_mags):
    fig2, ax2 = plt.subplots()
    fig2.suptitle('X_true')
    ax2.scatter(X_positions[:,0], X_positions[:,1],  c=Xtrue_mean_mags, cmap='RdBu')
    
def plot_hist(U_inlet_lst, analysis_inlet, truth_U_inlet, alpha_lst, analysis_alpha, truth_alpha):
    # # quick visual: histograms before vs after
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = 16
    axes[0].hist(U_inlet_lst, bins=bins, alpha=0.5, label='initial', color='blue')
    axes[0].hist(analysis_inlet, bins=bins, alpha=0.5, label='result', color='red')
    axes[0].axvline(truth_U_inlet, color='black', linestyle='--', label='truth')
    axes[0].set_xlabel('Inlet velocity')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].set_title('Inlet velocity: before vs after')
    
    axes[1].hist(alpha_lst, bins=bins, alpha=0.5, label='initial', color='blue')
    axes[1].hist(analysis_alpha, bins=bins, alpha=0.5, label='result', color='red')
    axes[1].axvline(truth_alpha, color='black', linestyle='--', label='truth')
    axes[1].set_xlabel('AoA')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].set_title('AoA: before vs after')
    
    plt.tight_layout()

def plot_field(X_positions, Xtrue_mean_mags, drone_xy_snapped):
    # print(np.min(Xtrue_mean_mags - X_mean_mags))
    # fig, ax = plt.subplots()
    # fig.suptitle("X_true minus X after ENKF")
    # sc = ax.scatter(X_positions[::2,0], X_positions[::2,1], c=Xtrue_mean_mags - X_mean_mags, cmap='RdBu',
    #                 vmin=np.min(Xtrue_mean_mags[:-2:] - X_mean_mags[:-2:]), vmax=0)
    
    # plt.colorbar(sc, ax=ax)
    
    fig2, ax2 = plt.subplots()
    fig2.suptitle('X_true')
    ax2.scatter(X_positions[::2,0], X_positions[::2,1],  c=Xtrue_mean_mags, cmap='RdBu')
    ax2.scatter(drone_xy_snapped[:,0], drone_xy_snapped[:,1], c='green')
    plt.show()

if __name__ == "__main__":
    for file in os.listdir('solutions'):
        print(file)    
        X = np.genfromtxt(f'solutions/{file}', delimiter=',', skip_header=True)
        X = X[:, 1:-1]
        print(X)
        X_positions = X[:, :2]
        Xtrue_mean_mags = np.linalg.norm(X[:, 2:], axis=1)
        print(Xtrue_mean_mags)
        plot_just_field(X_positions, Xtrue_mean_mags)
    
    plt.show()
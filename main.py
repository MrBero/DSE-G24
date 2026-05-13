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

def main(X_positions, X_ensemble, U_inlet_lst, alpha_lst, truth_U_inlet, truth_alpha, truth_path, point_distribution, dims, n_samples, plot=False):
    NOISE_STD = 0.15
    #point_distribution = 'optimized'
    seed = sum([ord(char) for char in 'PEACH_VIBE'])
    np.random.seed(seed)

    if point_distribution == 'random':
        drone_xy = sample_points(n_samples, (0, dims[0]), (-dims[1]/2, dims[1]/2), (107.5,0), truth_alpha, 
                                 100, 150, 50, 
                                 (107.5,0), 12.5)

    elif point_distribution == 'optimized':
        n_cells = (X_ensemble.shape[0] - 2) // 2
        xcoord = X_positions[0:2 * n_cells:2, 0]
        ycoord = X_positions[0:2 * n_cells:2, 1]
        drone_xy, score = pick_informative_drones(X_ensemble, xcoord, ycoord, NOISE_STD**2, n_samples, U_inf_row=-2, alpha_row=-1)

        # plt.figure(figsize=(10, 8))
        # # Use 'scatter' with your sorted arrays
        # # 's' controls point size, 'c' is the color depth, 'cmap' is the color theme
        # plt.scatter(xcoord, ycoord, c=score, s=10, cmap='cool', edgecolors='none')
        # plt.colorbar(label='Intensity Score')
        # plt.title("Point-based Heat Map")
        # plt.show()
    elif point_distribution == 'lines': 
        num_left = 5
        num_right = n_samples
        verts = rectangle_vertices(center=(107.5, 0), theta_deg=truth_alpha, s=100, a=150, h=50)
        left_boundary_pts = np.linspace(verts[0], verts[3], num_left)
        right_boundary_pts = np.linspace(verts[1], verts[2], num_right)
        drone_xy = np.vstack([left_boundary_pts, right_boundary_pts])
    
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
    if plot:
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.tight_layout()
        plot_just_field(axes[0], Xtrue_positions, Xtrue_mags, title = 'Truth', flat = False)
        plot_just_field(axes[1], X_positions, Xinitial_mean_mags, title = "Initial Means")
        plot_just_field(axes[2], X_positions, X_mean_mags, title = "Final Means")
        
        fig2 = plot_hist(U_inlet_lst, analysis_inlet, truth_U_inlet, alpha_lst, analysis_alpha, truth_alpha)
        fig3 = plot_error_field(X_positions, Xinitial_mean_mags, X_mean_mags, drone_xy)

        # plt.show()
        fig.savefig(f'results/U={truth_U_inlet}_a={truth_alpha}_{point_distribution.upper()}/truth-initial-final_U={truth_U_inlet}_a={truth_alpha}.png')
        fig2.savefig(f'results/U={truth_U_inlet}_a={truth_alpha}_{point_distribution.upper()}/boxplot_U={truth_U_inlet}_a={truth_alpha}.png')
        fig3.savefig(f'results/U={truth_U_inlet}_a={truth_alpha}_{point_distribution.upper()}/ENKF-change_U={truth_U_inlet}_a={truth_alpha}.png')
        print('Figures are saved.')
    return analysis_inlet.mean(), analysis_inlet.std(), analysis_alpha.mean(), analysis_alpha.std()
    
#load cfds
def load_main(directory):
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

    X_positions, X_ensemble, _, _ = load_all_cfds(path_lst, U_inlet_lst, alpha_lst)
    print('CFDs loaded!')
    return X_positions, X_ensemble, U_inlet_lst, alpha_lst

def rotmat(theta_deg):
    """2D rotation matrix for angle in degrees."""
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s],
                     [s,  c]])

def rectangle_vertices(center, theta_deg, s, a, h):
    """Calculates global (x, y) coordinates for the corners."""
    x0, y0 = center
    R = rotmat(theta_deg)
    corners_local = np.array([
        [-s, -h],
        [ a, -h],
        [ a,  h],
        [-s,  h],
    ])
    corners_global = corners_local @ R.T + np.array([x0, y0])
    return corners_global

def inside_sampling_region(x, y, center, theta_deg, s, a, h, excl_center, excl_radius):
    """Boolean mask for points inside rectangle and outside exclusion zone."""
    x0, y0 = center
    cx, cy = excl_center
    th = np.deg2rad(theta_deg)
    c, sn = np.cos(th), np.sin(th)
    dx, dy = x - x0, y - y0
    u =  dx * c + dy * sn
    v = -dx * sn + dy * c
    in_rect = (-s <= u) & (u <= a) & (-h <= v) & (v <= h)
    in_excl = (x - cx) ** 2 + (y - cy) ** 2 <= excl_radius ** 2
    return in_rect & (~in_excl)

def sample_points(n, xlim, ylim, center, theta_deg, s, a, h, excl_center, excl_radius, seed=42):
    """Rejection sampling for random interior points."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        xs = rng.uniform(xlim[0], xlim[1], size=5 * n)
        ys = rng.uniform(ylim[0], ylim[1], size=5 * n)
        mask = inside_sampling_region(xs, ys, center, theta_deg, s, a, h, excl_center, excl_radius)
        new_pts = np.column_stack([xs[mask], ys[mask]])
        pts.extend(new_pts.tolist())
    return np.array(pts[:n])

def run_single(truth_path, truth_U_inlet, truth_alpha, loaded, point_distribution, n_samples):
    inlet_mean, inlet_std, alpha_mean, alpha_std = main(*loaded,
                                                        truth_U_inlet=truth_U_inlet,
                                                        truth_alpha=truth_alpha,
                                                        truth_path = truth_path,
                                                        point_distribution = point_distribution,
                                                        dims = (800, 500),
                                                        n_samples = n_samples,
                                                        plot=True)


def run_sensitivity_study(truth_path, truth_U_inlet, truth_alpha, loaded):
    results = []
    with alive_bar(100) as bar:
        for n in np.arange(1, 101, 1):
            inlet_mean, inlet_std, alpha_mean, alpha_std = main(*loaded,
                                                                truth_U_inlet=truth_U_inlet,
                                                                truth_alpha=truth_alpha,
                                                                truth_path = truth_path,
                                                                point_distribution = 'random',
                                                                dims = (800, 500),
                                                                n_samples = n)
            results.append([n, inlet_mean, inlet_std, alpha_mean, alpha_std])
            bar()
    results = np.array(results)

    results_optimized = []
    with alive_bar(100) as bar:
        for n in np.arange(1, 101, 1):
            inlet_mean, inlet_std, alpha_mean, alpha_std = main(*loaded,
                                                                truth_U_inlet=truth_U_inlet,
                                                                truth_alpha=truth_alpha,
                                                                truth_path = truth_path,
                                                                point_distribution = 'optimized',
                                                                dims = (800, 500),
                                                                n_samples = n)
            results_optimized.append([n, inlet_mean, inlet_std, alpha_mean, alpha_std])
            bar()
    results_optimized = np.array(results_optimized)

    results_lines = []
    with alive_bar(100) as bar:
        for n in np.arange(1, 101, 1):
            inlet_mean, inlet_std, alpha_mean, alpha_std = main(*loaded,
                                                                truth_U_inlet=truth_U_inlet,
                                                                truth_alpha=truth_alpha,
                                                                truth_path = truth_path,
                                                                point_distribution = 'lines',
                                                                dims = (800, 500),
                                                                n_samples = n)
            results_lines.append([n, inlet_mean, inlet_std, alpha_mean, alpha_std])
            bar()
    results_lines = np.array(results_lines)

    fig, axes = plt.subplots(1,2)
    fig.suptitle('Sensitivity study: STD of initial conditions wrt. number of drones')

    axes[0].plot(results[:,0], results[:, 2], label='randomly within zone')
    axes[0].plot(results_optimized[:,0], results_optimized[:, 2], label='optimized selection')
    axes[0].plot(results_lines[:,0], results_lines[:, 2], label='line selection')
    axes[0].set_xlabel('Number of drones')
    axes[0].set_ylabel('Mean Initial Velocity STD')
    axes[0].set_title('Inlet velocity')

    axes[1].plot(results[:,0], results[:, 4], label='randomly within zone')
    axes[1].plot(results_optimized[:,0], results_optimized[:, 4], label='optimized selection')
    axes[1].plot(results_lines[:,0], results_lines[:, 4], label='line selection')
    axes[1].set_xlabel('Number of drones')
    axes[1].set_ylabel('AOA STD')
    axes[1].set_title('Alpha')

    plt.legend()
    # plt.show()

    np.savetxt(os.path.join('results',f'sensitivity_U={truth_U_inlet}_a={truth_alpha}', f'std-to-ndrones-sensitivity_U={truth_U_inlet}_a={truth_alpha}.csv'), results, header='n_drones, inlet-mean, inlet-std, alpha-mean, alpha-std')
    fig.savefig(os.path.join('results', f'sensitivity_U={truth_U_inlet}_a={truth_alpha}', f'std-to-ndrones-sensitivity_U={truth_U_inlet}_a={truth_alpha}.png'))

if __name__ == "__main__":
    case_folder = 'sim'
    directory = os.path.join(case_folder, 'CORRECTED_simulation_outputs')
    #truths 
    truth_path = os.path.join(case_folder, 'solution_data_truth14.5.csv')
    truth_U_inlet, truth_alpha = 14.5, -14.5
    point_distribution = 'lines'
    n_samples = 10
    loaded = load_main(directory)
    try: 
        os.mkdir(f'results/U={truth_U_inlet}_a={truth_alpha}_{point_distribution.upper()}')
    except: 
        pass
    run_single(truth_path, truth_U_inlet, truth_alpha, loaded, point_distribution=point_distribution, n_samples=n_samples)
    
    # try: 
    #     os.mkdir(f'results/sensitivity_U={truth_U_inlet}_a={truth_alpha}')
    # except: 
    #     pass
    # run_sensitivity_study(truth_path, truth_U_inlet, truth_alpha, loaded)


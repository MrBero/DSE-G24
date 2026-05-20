import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from load import load_one_cfd
import os

def make_drone_observations(truth_path, crude_path, drone_xy, noise_std, seed=42):
    """
    Sample a truth CFD at hand-picked drone locations, add noise,
    and return the values + their indices into X_ensemble.
    """
    rng = np.random.default_rng(seed)
    
    # load the truth case (same x, y, u, v format as ensemble files)
    truth_x, truth_y, truth_u, truth_v = load_one_cfd(truth_path)
    
    # build the same observation array structure as the original drone sampler:
    # columns are [x, y, u, v]
    obs = np.column_stack([truth_x, truth_y, truth_u, truth_v])
    
    # KDTree lookup 
    tree = KDTree(obs[:, :2])
    _, cell_indices = tree.query(drone_xy)
    measurements = obs[cell_indices, :]
    
    sample_file = os.path.join(crude_path, 'vel_13_aoa_-13', 'solution_data.csv') #TODO this has to become dynamic
    x, y, _, _ = load_one_cfd(sample_file)
    coords = np.column_stack([x, y])
    tree2 = KDTree(coords)
    _, actual_cell_indices = tree2.query(measurements[:, :2])

    print(actual_cell_indices.shape)
    y_predict = np.column_stack([coords[actual_cell_indices, :],  measurements[:, 2:]])

    # y_predict in the original style: snapped (x, y, u, v) at each drone
    
    # convert into the ETKF-compatible format
    n_drones = len(drone_xy)
    y_measured = np.zeros(2 * n_drones)
    obs_indices = np.zeros(2 * n_drones, dtype=int)
    
    for i, cell_idx in enumerate(actual_cell_indices):
        # u-measurement → row 2*cell_idx of X_ensemble
        y_measured[2*i]     = y_predict[i, 2] + rng.normal(0, noise_std)
        obs_indices[2*i]    = 2 * cell_idx
        
        # v-measurement → row 2*cell_idx + 1
        y_measured[2*i + 1] = y_predict[i, 3] + rng.normal(0, noise_std)
        obs_indices[2*i + 1] = 2 * cell_idx + 1
    
    drone_xy_snapped = y_predict[:, :2]
    
    return y_measured, obs_indices, drone_xy_snapped



""" 
#observation matrix
#load the matrix
observation = np.loadtxt('res.txt', skiprows=1, delimiter=',')

dims = (500, 200) #meters
obs = np.hstack([observation[:, 1:4], observation[:, 7:]])
print(obs.shape)

n_samples = 100
sample_points = np.vstack([np.random.random(n_samples) * dims[0], 
                           np.random.random(n_samples) * dims[1] - dims[1]/2]).T
print(sample_points.shape)

tree = KDTree(obs[:,:2])
_, index = tree.query(sample_points)
# print(index)

y_predict = obs[index, :]
# print(y_predict)


plt.scatter(obs[:,0], obs[:,1], c=obs[:,3], cmap='RdBu')
# plt.scatter(sample_points[:,0], sample_points[:,1], c='red')
plt.scatter(y_predict[:,0], y_predict[:,1], c=y_predict[:,3])
plt.colorbar()
plt.show()
#sample the complete matrix with a bunch of points in space

"""
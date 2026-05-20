import numpy as np
from load import load_all_cfds


def pick_informative_drones(X_ensemble, cell_x, cell_y, R, n_drones,
                            U_inf_row=-2, alpha_row=-1):
    """
    Pick the n_drones cell locations that, if measured, would reduce
    the variance in the parameters (U_inf and alpha) the most.
    
    Uses the independent-measurement approximation: each measurement
    (u or v at a cell) is treated as contributing independently.
    This slightly overestimates information when u and v at the same
    cell are correlated, but the ranking is essentially unaffected.
    
    Assumes the X_ensemble layout from load.py:
        rows 0, 2, 4, ...  -> u at cells 0, 1, 2, ...   (even rows)
        rows 1, 3, 5, ...  -> v at cells 0, 1, 2, ...   (odd rows)
        row -2             -> U_inf
        row -1             -> alpha
    
    Parameters
    ----------
    X_ensemble : (n_state, N) ensemble matrix from load_all_cfds
    cell_x, cell_y : (n_cells,) coordinates of each mesh cell
    n_cells : number of mesh cells
    R : measurement noise variance (scalar, same for u and v sensors)
    n_drones : how many drones to place
    U_inf_row, alpha_row : row indices of the parameters (defaults match load.py)
    
    Returns
    -------
    drone_xy : (n_drones, 2) array of (x, y) positions, ready to feed into
               make_drone_observations
    score    : (n_cells,) array of information score per cell, useful for
               visualisation
    """
    # ensemble mean and deviations - we work through delta_X so we never
    # build the full covariance matrix.
    '''
    Comment Seppe: We can use these results in the etkf.py so that we dont calculate it twice
    '''
    N = X_ensemble.shape[1]                                 #amount of cfds
    x_mean = X_ensemble.mean(axis=1, keepdims=True)         #means of the row
    delta_X = X_ensemble - x_mean                           #delta X
    n_cells = cell_x.shape[0]


    # extract parameter deviations (one number per ensemble member)
    delta_Uinf  = delta_X[U_inf_row, :]    # (N,)
    delta_alpha = delta_X[alpha_row, :]    # (N,)
    
    # extract flow deviations using INTERLEAVED slicing
    # rows 0, 2, 4, ... -> u-rows for cells 0, 1, 2, ...
    # rows 1, 3, 5, ... -> v-rows for cells 0, 1, 2, ...
    # this matches the layout produced by load.py's load_all_cfds.
    delta_u = delta_X[0:2*n_cells:2, :]    # (n_cells, N)
    delta_v = delta_X[1:2*n_cells:2, :]    # (n_cells, N)
    
    # covariances between each parameter and each cell's u/v.
    # broadcasting: delta_u is (n_cells, N), delta_Uinf is (N,).
    # element-wise multiply then sum across ensemble members.
    cov_Uinf_u  = (delta_u * delta_Uinf).sum(axis=1) / (N - 1)   # (n_cells,)
    cov_Uinf_v  = (delta_v * delta_Uinf).sum(axis=1) / (N - 1)      # [[delv0,1 * delUinf_1; delv0,2 * delUinf_2; ...].sum(axis=1)
                                                                    #   [delv1,1 * delUinf_1; delv1,2 * delUinf_2, ...]].sum(axis=1)
    cov_alpha_u = (delta_u * delta_alpha).sum(axis=1) / (N - 1)
    cov_alpha_v = (delta_v * delta_alpha).sum(axis=1) / (N - 1)
    
    # variance of u and v at every cell
    var_u = (delta_u ** 2).sum(axis=1) / (N - 1)                 # (n_cells,)
    var_v = (delta_v ** 2).sum(axis=1) / (N - 1)
    
    # prior variances of the parameters (scalars, for normalisation)
    var_Uinf  = (delta_Uinf  ** 2).sum() / (N - 1)
    var_alpha = (delta_alpha ** 2).sum() / (N - 1)
    
    # variance reduction per cell, per parameter.
    # for cell i, measuring u_i decreases var(U_inf) by
    #   [cov(U_inf, u_i)]^2 / (var(u_i) + R)
    # under the independent-measurement approximation, the contributions
    # from u and v at the same cell simply add.
    reduction_Uinf  = (cov_Uinf_u  ** 2) / (var_u + R) + (cov_Uinf_v  ** 2) / (var_v + R)
    reduction_alpha = (cov_alpha_u ** 2) / (var_u + R) + (cov_alpha_v ** 2) / (var_v + R)
    
    # combine into a single dimensionless score.
    # each term is the fraction of that parameter's variance reduced,
    # so they can be added directly regardless of physical units.
    score = reduction_Uinf / var_Uinf + reduction_alpha / var_alpha

    # sort all cell indices by score, highest first
    sorted_idx = np.argsort(score)[::-1]

    min_dist = 0.0
    selected = []  # list of (x, y) tuples

    for idx in sorted_idx:
        if len(selected) == n_drones:
            break

        x, y = cell_x[idx], cell_y[idx]

        # check distance to all already-selected drones
        if selected:
            sel_arr = np.asarray(selected)
            dists = np.hypot(sel_arr[:, 0] - x, sel_arr[:, 1] - y)
            if np.any(dists < min_dist):
                continue  # too close, skip this cell

        selected.append((x, y))


    drone_xy = np.asarray(selected) 
    return drone_xy, score


if __name__ == "__main__":
    import os
    
    FOLDER = "fakedata"
    N = 12
    R = 0.15 ** 2
    
    path_lst = [os.path.join(FOLDER, f"member_{i:02d}.csv") for i in range(N)]
    U_inlet_lst = np.load(os.path.join(FOLDER, "U_inlet_lst.npy"))
    alpha_lst   = np.load(os.path.join(FOLDER, "alpha_lst.npy"))
    M, X_ensemble = load_all_cfds(path_lst, U_inlet_lst, alpha_lst)
    
    # number of cells - subtract 2 parameter rows, then halve (u + v per cell)
    n_cells = (X_ensemble.shape[0] - 2) // 2
    
    # cell coords sit in columns 0, 1 of M, on the u-rows (interleaved layout)
    cell_x = M[0:2*n_cells:2, 0]
    cell_y = M[0:2*n_cells:2, 1]
    
    drone_xy, score = pick_informative_drones(
        X_ensemble, cell_x, cell_y, n_cells, R, n_drones=8
    )
    
    print("Selected drone positions:")
    for i, (x, y) in enumerate(drone_xy):
        print(f"  drone {i}: ({x:.2f}, {y:.2f})")
    print(f"Score range across all cells: [{score.min():.4f}, {score.max():.4f}]")
import numpy as np


def load_one_cfd(csv_path):
    """
    Load one CFD output file with columns x, y, u, v.
    Returns four 1D NumPy arrays: x, y, u, v.
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    
    # sort by x, then by y, for consistent ordering across files
    sort_idx = np.lexsort((data[:, 1], data[:, 0]))
    data = data[sort_idx]
    
    x = data[:, 1]
    y = data[:, 2]
    u = data[:, 3]
    v = data[:, 4]
    
    return x, y, u, v


def load_all_cfds(path_lst, U_inlet_lst, alpha_lst):
    """
    Layout of matrix M:
        - shape: (2 * n_cells + 2, 2 + N)
        - columns 0 and 1 are x and y coordinates
        - columns 2 to 2+N-1 are the ensemble values
        - rows come in pairs: row 2k is u at cell k, row 2k+1 is v at cell k
        - last two rows are inlet velocity and AoA, with NaN for x, y
    """
    # assert len(path_lst) == len(U_inlet_lst) == len(alpha_lst), \
    #     "path list, U_inlet list, and alpha list must have the same length"
    
    N = len(path_lst) #amount of cfds run
    
    # load the first file to get mesh size and reference coordinates
    x_ref, y_ref, u_ref, v_ref = load_one_cfd(path_lst[0])
    n_cells = len(x_ref) #amount of gridpoints
    
    # allocate the matrix
    M = np.zeros((2 * n_cells + 2, 2 + N)) #get shape ready of matrix M
    
    # fill the first two columns (x and y)
    M[0:2*n_cells:2, 0] = x_ref       # u-rows get x_ref
    M[0:2*n_cells:2, 1] = y_ref       # u-rows get y_ref
    M[1:2*n_cells:2, 0] = x_ref       # v-rows get x_ref
    M[1:2*n_cells:2, 1] = y_ref       # v-rows get y_ref
    
    # last two rows (inlet velocity and angle) have no x, y 
    M[-2, 0] = np.nan
    M[-2, 1] = np.nan
    M[-1, 0] = np.nan
    M[-1, 1] = np.nan
    
    # fill member 0 (already loaded)
    M[0:2*n_cells:2, 2] = u_ref
    M[1:2*n_cells:2, 2] = v_ref
    M[-2, 2] = U_inlet_lst[0]
    M[-1, 2] = alpha_lst[0]
    
    # loop over the rest of the members
    for i in range(1, N):
        x, y, u, v = load_one_cfd(path_lst[i])
        
        # mesh consistency check
        assert np.allclose(x, x_ref), \
            f"mesh x-coords differ between {path_lst[0]} and {path_lst[i]}"
        assert np.allclose(y, y_ref), \
            f"mesh y-coords differ between {path_lst[0]} and {path_lst[i]}"
        
        # column index for this member: i + 2 (since cols 0 and 1 are x, y)
        col = i + 2
        M[0:2*n_cells:2, col] = u
        M[1:2*n_cells:2, col] = v
        M[-2, col] = U_inlet_lst[i]
        M[-1, col] = alpha_lst[i]
    
    # M_data is just M without the first two columns
    M_data = M[:, 2:]
    M_drones = M[:,:2]
    print(f'Coords shape: {M_drones.shape}, States shape: {M_data.shape}')
    return M_drones, M_data, x_ref, y_ref


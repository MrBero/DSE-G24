import numpy as np
from load import load_all_cfds


def pick_informative_drones(X_ensemble, cell_x, cell_y, R, n_drones,
                            U_inf_row=-2, alpha_row=-1,
                            min_dist=0.0):
    """
    Sequentially pick n_drones cell locations that maximally reduce the
    posterior variance of the parameters (U_inf and alpha).

    Improvements over the one-shot ranking version:

    1. SEQUENTIAL greedy selection.
       After each pick, the ensemble deviation matrix delta_X is updated as
       if a measurement had been taken at that cell, using the ETKF
       deviation update (delta_X^a = delta_X^f @ T). The next score then
       reflects what is STILL uncertain, so redundant nearby locations are
       penalised automatically — no geometric spacing rule needed.

    2. JOINT 2x2 variance reduction at every cell.
       Does NOT assume u and v at the same cell are independent. Cov(u, v)
       enters through the inverse of the 2x2 innovation block, so a cell
       with strongly coupled u and v is correctly de-weighted relative to a
       cell where they carry independent information.

    3. RE-NORMALISED score per iteration.
       The two parameter terms are normalised by the CURRENT remaining
       variance of each parameter, not the initial variance. Once U_inf
       has been mostly constrained by early drones, the U_inf term shrinks
       relative to the alpha term, and the algorithm naturally re-balances
       toward alpha. This is what fixes the failure mode where the old
       score (normalised by initial variance only) was dominated by U_inf
       and never paid attention to alpha.

    4. NO full covariance matrix.
       Everything is done through delta_X. Cell-wise statistics are O(N)
       per cell, the deviation update is O(n_state * N^2) per pick.

    Parameters
    ----------
    X_ensemble : (n_state, N) ensemble matrix from load_all_cfds
    cell_x, cell_y : (n_cells,) coordinates of each mesh cell
    R : measurement noise variance (scalar; same for u and v sensors)
    n_drones : number of drones to place
    U_inf_row, alpha_row : row indices of the parameters in X_ensemble
    min_dist : minimum allowed spacing (m) between selected drones.
               With sequential update, redundancy is already handled, so
               default is 0. Set this only if there is a physical /
               operational reason (drone separation, airspace, etc.).

    Returns
    -------
    drone_xy : (n_drones, 2) array of (x, y) positions, in selection order
    score    : (n_cells,) score at the final iteration, useful for plotting
    """
    N = X_ensemble.shape[1]
    n_cells = cell_x.shape[0]

    # work on a copy because we mutate delta_X sequentially
    x_mean = X_ensemble.mean(axis=1, keepdims=True)
    delta_X = (X_ensemble - x_mean).copy()

    selected_indices = []
    selected_xy = []
    score_last = np.zeros(n_cells)

    for k in range(n_drones):
        # --- 1. extract CURRENT parameter and flow deviations -----------
        delta_Uinf  = delta_X[U_inf_row, :]            # (N,)
        delta_alpha = delta_X[alpha_row, :]            # (N,)
        delta_u = delta_X[0:2 * n_cells:2, :]          # (n_cells, N)
        delta_v = delta_X[1:2 * n_cells:2, :]          # (n_cells, N)

        # --- 2. cell-wise statistics from delta_X (no covariance matrix) ---
        # parameter-flow covariances
        cov_Uinf_u  = (delta_u * delta_Uinf).sum(axis=1)  / (N - 1)   # (n_cells,)
        cov_Uinf_v  = (delta_v * delta_Uinf).sum(axis=1)  / (N - 1)
        cov_alpha_u = (delta_u * delta_alpha).sum(axis=1) / (N - 1)
        cov_alpha_v = (delta_v * delta_alpha).sum(axis=1) / (N - 1)

        # innovation 2x2 block per cell:  S_i = Cov([u_i, v_i]) + R*I_2
        a = (delta_u ** 2).sum(axis=1) / (N - 1) + R      # (n_cells,)
        d = (delta_v ** 2).sum(axis=1) / (N - 1) + R
        b = (delta_u * delta_v).sum(axis=1) / (N - 1)     # cross term — NOT zero

        det_S = a * d - b ** 2                            # (n_cells,)
        safe  = det_S > 1e-12                             # numerical floor

        # --- 3. variance reduction via the 2x2 quadratic form ---------
        # Delta Var(theta) = -c^T S^-1 c
        #                  = -(c_u^2 * d - 2 c_u c_v b + c_v^2 * a) / det_S
        # we return the POSITIVE reduction (i.e. -Delta Var, always >= 0).
        def quad_reduction(c_u, c_v):
            num = c_u ** 2 * d - 2.0 * c_u * c_v * b + c_v ** 2 * a
            red = np.zeros_like(num)
            red[safe] = num[safe] / det_S[safe]
            # guard against tiny negative values from FP cancellation
            return np.maximum(red, 0.0)

        reduction_Uinf  = quad_reduction(cov_Uinf_u,  cov_Uinf_v)
        reduction_alpha = quad_reduction(cov_alpha_u, cov_alpha_v)

        # --- 4. normalise by CURRENT (not initial) parameter variances --
        var_Uinf  = max((delta_Uinf  ** 2).sum() / (N - 1), 1e-12)
        var_alpha = max((delta_alpha ** 2).sum() / (N - 1), 1e-12)
        score = reduction_Uinf / var_Uinf + reduction_alpha / var_alpha
        score_last = score.copy()

        # --- 5. exclude already-selected and (optionally) min_dist viol. -
        if selected_indices:
            score[np.asarray(selected_indices)] = -np.inf
        if min_dist > 0.0 and selected_xy:
            sel = np.asarray(selected_xy)
            dx = cell_x[:, None] - sel[:, 0][None, :]
            dy = cell_y[:, None] - sel[:, 1][None, :]
            too_close = np.any(np.hypot(dx, dy) < min_dist, axis=1)
            score = np.where(too_close, -np.inf, score)

        # --- 6. pick best cell ------------------------------------------
        best = int(np.argmax(score))
        selected_indices.append(best)
        selected_xy.append((cell_x[best], cell_y[best]))

        # --- 7. ETKF deviation update for THIS single measurement -------
        # we hypothetically measure (u, v) at cell `best` with noise R*I_2.
        # delta_y = delta_X[[2*best, 2*best+1], :]  has shape (2, N).
        # Standard ETKF transform:
        #   M = I_N + delta_y^T R^-1 delta_y / (N-1)
        #   T = M^(-1/2)         (via eigendecomp; M is symmetric PSD)
        #   delta_X <- delta_X @ T
        # Skip on the last iteration — we won't use delta_X again.
        if k < n_drones - 1:
            delta_y = delta_X[[2 * best, 2 * best + 1], :]            # (2, N)
            # R is scalar (same variance for u and v sensors), so R^-1 = 1/R * I
            M = np.eye(N) + (delta_y.T @ delta_y) / ((N - 1) * R)     # (N, N)
            eigvals, Z = np.linalg.eigh(M)
            eigvals = np.maximum(eigvals, 1e-12)
            T = (Z * (1.0 / np.sqrt(eigvals))) @ Z.T                  # (N, N)
            delta_X = delta_X @ T

    return np.asarray(selected_xy), score_last


if __name__ == "__main__":
    import os

    FOLDER = "fakedata"
    N = 12
    R = 0.15 ** 2

    path_lst = [os.path.join(FOLDER, f"member_{i:02d}.csv") for i in range(N)]
    U_inlet_lst = np.load(os.path.join(FOLDER, "U_inlet_lst.npy"))
    alpha_lst   = np.load(os.path.join(FOLDER, "alpha_lst.npy"))
    M, X_ensemble = load_all_cfds(path_lst, U_inlet_lst, alpha_lst)

    n_cells = (X_ensemble.shape[0] - 2) // 2
    cell_x = M[0:2 * n_cells:2, 0]
    cell_y = M[0:2 * n_cells:2, 1]

    drone_xy, score = pick_informative_drones(
        X_ensemble, cell_x, cell_y, R, n_drones=8
    )

    print("Selected drone positions:")
    for i, (x, y) in enumerate(drone_xy):
        print(f"  drone {i}: ({x:.2f}, {y:.2f})")
    print(f"Score range at final iter: [{score.min():.4f}, {score.max():.4f}]")

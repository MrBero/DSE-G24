import numpy as np


def etkf(X_ensemble, y_measured, y_pred, R_sensor):
    """
    Ensemble Transform Kalman Filter update.

    X_ensemble : (n_state, N) each column is one ensemble member's state vector
    y_measured : (m,)         observation vector (real measurements)
    y_pred     : (m, N)       predicted observations for each ensemble member
                                (computed externally, e.g. by slicing X_ensemble
                                at the rows that correspond to sensor locations)
    R_sensor   : (m,)         diagonal of observation noise covariance (variances)

    Returns X_analysis of shape (n_state, N).
    """

    # shape checks
    n_state, N = X_ensemble.shape
    m = y_measured.shape[0]

    if y_pred.shape != (m, N):
        print(f"y_pred has shape {y_pred.shape}, expected ({m}, {N})")
    if R_sensor.shape != (m,):
        print(f"R_sensor has shape {R_sensor.shape}, expected ({m},)")

    # ensemble mean and deviations
    print('ensemble mean...')
    x_mean = X_ensemble.mean(axis=1, keepdims=True)
    dX = X_ensemble - x_mean

    # predicted observation mean and deviations
    print('predicted mean...')
    y_pred_mean = y_pred.mean(axis=1, keepdims=True)
    dy_pred = y_pred - y_pred_mean

    print('eq 17...')
    # eq 17: build M = I + dX^T H^T [(N-1)R]^-1 H dX
    R_inv_dy = dy_pred / R_sensor[:, None]
    M = np.eye(N) + (dy_pred.T @ R_inv_dy) / (N - 1)

    print('eigs...')
    # eigendecomposition of M (symmetric, so use eigh)
    eigvals, Z = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-12)  # guard against tiny negative floats

    print('eq 19...')
    # eq 19: T = Z Sigma^-1/2 Z^T
    T = (Z * (1.0 / np.sqrt(eigvals))) @ Z.T

    print('eq 18 and 20...')
    # eq 18 + 20: mean update via the Kalman gain
    # K_hat applied to innovation gives: K_t*d = delX @ Z @ eigenval-1 @ ZT (delX)T HT [(N-1)R]-1 * d
    innovation = y_measured[:, None] - y_pred_mean # = d
    R_inv_innovation = innovation / ((N - 1) * R_sensor[:, None]) # [(N-1)R]-1 * d
    rhs = dy_pred.T @ R_inv_innovation # (delX)T HT [(N-1)R]-1 * d
    w = Z @ ((Z.T @ rhs) / eigvals[:, None])  # Z @ eigenval-1 @ ZT (delX)T HT [(N-1)R]-1 * d
    mean_corr = dX @ w
    x_mean_analysis = x_mean + mean_corr

    print('eq 21...')
    # eq 21: deviation update
    dX_analysis = dX @ T

    print('eq 22...')
    # eq 22: reassemble
    X_analysis = x_mean_analysis + dX_analysis

    return X_analysis, dX_analysis, mean_corr


if __name__ == "__main__":
    # toy test: 7-state (3 cells x 2 components + 1 parameter), 3 members
    X = np.array([
        [4.2, 5.1, 6.0],   # u at cell 1
        [3.8, 4.7, 5.6],   # u at cell 2
        [4.0, 4.9, 5.8],   # u at cell 3
        [0.1, 0.2, 0.3],   # v at cell 1
        [-0.3, -0.4, -0.5],# v at cell 2
        [0.0, 0.1, 0.1],   # v at cell 3
        [4.5, 5.5, 6.5],   # inlet velocity (the parameter)
    ])

    # measure u at cell 1; truth-ish value 5.3
    # row 0 of X_ensemble corresponds to "u at cell 1"
    obs_indices = np.array([0])
    y_pred = X[obs_indices, :]   # shape (1, 3)
    
    y = np.array([5.3])
    R = np.array([0.1])

    X_a = etkf(X, y, y_pred, R)

    print("Forecast inlet velocities:", X[-1, :])
    print(f"Forecast mean inlet: {X[-1, :].mean():.3f}")
    print("Analysis inlet velocities:", X_a[-1, :])
    print(f"Analysis mean inlet: {X_a[-1, :].mean():.3f}")
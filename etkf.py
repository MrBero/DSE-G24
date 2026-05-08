import numpy as np


def etkf(X_ensemble, y_measured, H_obs, R_sensor):
    """
    Ensemble Transform Kalman Filter update.

    X_ensemble : (n_state, N) — each column is one ensemble member's state vector
    y_measured : (m,)         — observation vector
    H_obs      : (m, n_state) — observation operator
    R_sensor   : (m,)         — diagonal of observation noise covariance (variances)

    Returns X_analysis of shape (n_state, N).
    """

    # shape checks
    n_state, N = X_ensemble.shape
    m = y_measured.shape[0]

    if H_obs.shape != (m, n_state):
        print(f"H_obs has shape {H_obs.shape}, expected ({m}, {n_state})")
    if R_sensor.shape != (m,):
        print(f"R_sensor has shape {R_sensor.shape}, expected ({m},)")

    # ensemble mean and deviations
    x_mean = X_ensemble.mean(axis=1, keepdims=True)
    dX = X_ensemble - x_mean

    # predicted observations and their deviations
    y_pred = H_obs @ X_ensemble
    y_pred_mean = y_pred.mean(axis=1, keepdims=True)
    dy_pred = y_pred - y_pred_mean

    # eq 17: build M = I + dX^T H^T [(N-1)R]^-1 H dX
    R_inv_dy = dy_pred / R_sensor[:, None]
    M = np.eye(N) + (dy_pred.T @ R_inv_dy) / (N - 1)

    # eigendecomposition of M (symmetric, so use eigh)
    eigvals, Z = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-12)  # guard against tiny negative floats

    # eq 19: T = Z Sigma^-1/2 Z^T
    T = (Z * (1.0 / np.sqrt(eigvals))) @ Z.T

    # eq 18 + 20: mean update via the Kalman gain
    # K_hat applied to innovation gives: K_t*d = delX @ Z @ eigenval-1 @ ZT (delX)T HT [(N-1)R]-1 * d
    innovation = y_measured[:, None] - y_pred_mean #=d
    R_inv_innovation = innovation / ((N - 1) * R_sensor[:, None]) #[(N-1)R]-1 * d, R is diagonal so R^-1 is just elemntwise division
    rhs = dy_pred.T @ R_inv_innovation # (delX)T HT [(N-1)R]-1 * d
    w = Z @ ((Z.T @ rhs) / eigvals[:, None])  #  Z @ eigenval-1 @ ZT (delX)T HT [(N-1)R]-1 * d
    mean_corr = dX @ w
    x_mean_analysis = x_mean + mean_corr

    # eq 21: deviation update
    dX_analysis = dX @ T

    # eq 22: reassemble
    X_analysis = x_mean_analysis + dX_analysis

    return X_analysis


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
    H = np.zeros((1, 7))
    H[0, 0] = 1.0
    y = np.array([5.3])
    R = np.array([0.1])

    X_a = etkf(X, y, H, R)

    print("Forecast inlet velocities:", X[-1, :])
    print(f"Forecast mean inlet: {X[-1, :].mean():.3f}")
    print("Analysis inlet velocities:", X_a[-1, :])
    print(f"Analysis mean inlet: {X_a[-1, :].mean():.3f}")
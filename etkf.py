

import numpy as np


def etkf(X_ensemble, y_measured, H_obs, R_sensor):
    """
    X_ensemble has dimensions: (n_states x N) with  n_states being total numbers we're estimating, e.g. points * properties/point + unsure initial conditions
                                                    N being the amount of ensembles
    y_measured: (m, )                              m being the amount of observations
    H: (m, n_state)
    """
    #check for shapes

    n_state, N = X_ensemble.shape
    m = y_measured.shape[0]

    if H_obs.shape != (m, n_state):
        print(f"The observation operator does not have the expected shape ({m}, {n_state })" )

    if R_sensor.shape != (m,):
        print(f"The observation variance matrix does not have the expected shape ({m, })")

    #get mean of X and deviation dX
    x_mean = X_ensemble.mean(axis=1, keepdims=True) #Is this gonna include the inlet velocity in the mean??
    dX = X_ensemble - x_mean

    #get predicted observation HX
    y_pred = H_obs @ X_ensemble

    #get mean of y_pred and deviation of y_pred
    y_pred_mean = y_pred.mean(axis=1, keepdims=True)
    dy_pred = y_pred - y_pred_mean

    #eq 17
    R_inv_dy = dy_pred / R_sensor[:, None]
    M = np.eye(N) + (dy_pred.T @ R_inv_dy) / (N-1)

    #get eigenvectors from M as it is symmetric
    eigvals, Z = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-12) #to get rid of negative floating point eigvals
    
    #Compute T, eq 19
    T = (Z * (1.0 / np.sqrt(eigvals)))  @ Z.T  

    #calculate weights
    innovation = y_measured[:, None] - y_pred_mean
    R_inv_innovation = innovation / R_sensor[:, None]
    rhs = dy_pred.T @ R_inv_innovation
    w = Z @ ((Z.T @ rhs) / eigvals[:, None]) # using M^-1 = Z eig^-1 ZT 

    #update mean
    mean_corr = dX @ w
    x_mean_analysis = x_mean + mean_corr

    #use eq21 fo update deviation
    dX_analysis = dX @ T

    #use eq 22 to reassemble
    X_analysis = x_mean_analysis + dX_analysis

    return X_analysis

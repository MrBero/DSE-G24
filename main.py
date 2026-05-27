import numpy as np
import scipy.differentiate as scdiff
import matplotlib.pyplot as plt
from sampling import sample
import time

training_point_n = 100
ground_truth = sample('inputs/Field.csv', 'inputs/wall.csv', method='random', num_samples=training_point_n)
# ground_coords = ground_truth[]
ground_coords = ground_truth[['x-target', 'y-target', 'z-target']].to_numpy()
print(ground_truth.shape)

# def matern_five_two_isotropic(a, b, ell, var):
#     dist = np.abs(a - b)
#     term_1 = 1 + (np.sqrt(5) * dist) / ell + (5 * (dist**2)) / (3 * (ell**2))
#     term_2 = (-(np.sqrt(5) * dist)) / ell
#     term_3 = np.exp(term_2)
#     val = var * term_1 * term_3
#     return val

# def matern_five_two_anisotropic_2D(x_1, x_2, y_1, y_2, ell_1, ell_2, var):
#     r_ARD = np.sqrt(((x_1 - y_1)**2 / ell_1**2) + ((x_2 - y_2)**2 / ell_2**2))
#     term_1 = 1 + np.sqrt(5)*r_ARD + (5/3)*(r_ARD**2)
#     term_2 = np.exp(-np.sqrt(5)*r_ARD)
#     val = var * term_1 * term_2
#     return val

def matern52_np(v1, v2, ell_1=1.0, ell_2=2.0, ell_3=1.5, var=1.0):
    diff = v2 - v1
    r = np.sqrt((diff[0]/ell_1)**2 + (diff[1]/ell_2)**2 + (diff[2]/ell_3)**2)
    return var * (1 + np.sqrt(5)*r + (5/3)*r**2) * np.exp(-np.sqrt(5)*r)

def numerical_hessian(f, x, eps=1e-4):
    """Central-difference Hessian of scalar f: R^n -> R"""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i] = ej[j] = eps
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej)
                      - f(x - ei + ej) + f(x - ei - ej)) / (4 * eps**2)
            H[j, i] = H[i, j]
    return H

def Hemholtz_K0(V1, V2):
    H = numerical_hessian(lambda u: matern52_np(u, V2), V1)
    return np.block([[-H[1,1]-H[2,2], H[0,1], H[0,2]],
                    [H[1,0], -H[2,2]-H[0,0], -H[1,2]],
                    [H[2,0], H[0,1], -H[0,0]-H[1,1]]])

K_matrix = np.zeros((training_point_n*3, training_point_n*3))

sigma_noise = 0.05
variance = 1
for i in range(training_point_n):
    for j in range(training_point_n):
            if i==j:
                K_matrix[3*i:3*i+3, 3*j:3*j+3] += sigma_noise**2 #terms on the diagonals
            p_1 = ground_coords[i,:]
            p_2 = ground_coords[j,:]
            K_matrix[3*i:3*i+3, 3*j:3*j+3] += Hemholtz_K0(p_1, p_2)
print(K_matrix.shape)
#invert the noised K matrix
# tick = time.thread_time()
# K_noised_inv = np.linalg.inv(K_matrix)
# tock = time.thread_time()
# print(K_noised_inv)
# print(f'Inversion complete in {tock-tick:.3f}s')


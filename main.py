import numpy as np
import matplotlib.pyplot as plt
from sampling import sample
import time

#ground truth values or real world measurements; 'training' for GPR
training_point_n = 100
ground_truth, bounds, wall_df = sample('inputs/Field.csv', 'inputs/wall.csv', method='random', num_samples=training_point_n)
training_coords = ground_truth[['x-target', 'y-target', 'z-target']].to_numpy() #leave unflattened for functionality
training_vels = ground_truth[['x-velocity','y-velocity','z-velocity']].to_numpy().reshape(-1,1) #flatten to match dims in equation 1.7
# print(training_vels.shape)


#test points to points at which we seek GPR to evaluate the field
res = 10
x,y,z = np.meshgrid(np.linspace(bounds[0,0], bounds[0,1], res), 
                        np.linspace(bounds[1,0], bounds[1,1], res), 
                        np.linspace(bounds[2,0], bounds[2,1], res),
                        indexing='ij')

test_points = np.stack([x,y,z], axis=-1).reshape(-1,3)
test_point_n = test_points.shape[0]

print(f"Total number of training points: {training_point_n}\nTotal number of test points: {test_point_n}")

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
                    [H[2,0], H[2,1], -H[0,0]-H[1,1]]])


def assemble_dat_shi(points_1, points_2, noise=True): #assemble a matrix of covariances (3x3 matrix per covariance calc) given 2 sets of points
    n_1 = points_1.shape[0]
    n_2 = points_2.shape[0]
    print(n_1, n_2)
    result_matrix = np.zeros((n_1*3, n_2*3))
    sigma_noise = 0.05
    for i in range(n_1):
        for j in range(n_2):
                if i==j and noise:
                    result_matrix[3*i:3*i+3, 3*j:3*j+3] += sigma_noise**2 * np.eye(3) #terms on the diagonals
                p_1 = points_1[i,:]
                p_2 = points_2[j,:]
                result_matrix[3*i:3*i+3, 3*j:3*j+3] += Hemholtz_K0(p_1, p_2)
    print(result_matrix.shape)
    return result_matrix

tick = time.thread_time()
K_matrix = assemble_dat_shi(training_coords, training_coords)
tock = time.thread_time()
print(f'K_matrix assembled in {tock-tick:.3f}s')
print(K_matrix.shape)

#invert the noised K matrix
tick = time.thread_time()
K_noised_inv = np.linalg.inv(K_matrix)
tock = time.thread_time()
print(f'Inversion complete in {tock-tick:.3f}s')
print(K_noised_inv.shape)

tick = time.thread_time()
k_star = assemble_dat_shi(test_points, training_coords)
tock = time.thread_time()
print(f'K_star assembled in {tock-tick:.3f}s')
print(k_star.shape)

means_training = np.zeros((training_point_n * 3,1))
means_tests = np.zeros((test_point_n * 3, 1))

print((k_star @ K_noised_inv).shape)
print((training_vels - means_training).shape)

tick = time.thread_time()
GPR_posterior = means_tests + k_star @ K_noised_inv @ (training_vels - means_training)
tock = time.thread_time()
print(f'GPR Posterior generated in {tock-tick:.3f}s')
print(GPR_posterior.shape)

GPR_posterior_reshaped = GPR_posterior.reshape(-1,3)

ax = plt.figure().add_subplot(projection='3d')
ax.scatter3D(wall_df[['x-coordinate']].values, wall_df[['y-coordinate']].values, wall_df[['z-coordinate']].values, c='black')
ax.quiver(test_points[:,0],test_points[:,1],test_points[:,2],
          GPR_posterior_reshaped[:,0],GPR_posterior_reshaped[:,1],GPR_posterior_reshaped[:,2],
          length=0.01, normalize=True)

plt.show()
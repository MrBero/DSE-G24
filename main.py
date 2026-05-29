import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['JAX_PLATFORMS'] = 'cpu' #if jax CUDA is installed, then we force cpu in this case. comment if gpu compute is preferred
import time
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
import jax.scipy.optimize
jax.config.update("jax_enable_x64", True)
import scipy.interpolate

from potential_flow_run import PotentialFlowSolver
from sampling import sample

def matern52_np(v1, v2, ell, var):
    diff = v2 - v1
    r = jnp.sqrt((diff[0]/ell[0])**2 + (diff[1]/ell[1])**2 + (diff[2]/ell[2])**2 + 1e-8)
    return var * (1 + jnp.sqrt(5)*r + (5/3)*r**2) * jnp.exp(-jnp.sqrt(5)*r)

@jax.jit
def Hemholtz_K0(V1, V2, ell, var):
    # H = jax.hessian(lambda u: matern52_np(u, V2))
    H = jax.hessian(matern52_np)(V1, V2, ell, var)
    return jnp.array([[-H[1,1]-H[2,2], H[0,1], H[0,2]],
                    [H[1,0], -H[2,2]-H[0,0], H[1,2]],
                    [H[2,0], H[2,1], -H[0,0]-H[1,1]]])

def assemble_dat_shi(points_1, points_2, ell, var, noise=True): #assemble a matrix of covariances (3x3 matrix per covariance calc) given 2 sets of points
    n_1 = points_1.shape[0]
    n_2 = points_2.shape[0]
    blocks = jax.vmap(lambda a: jax.vmap(lambda b: Hemholtz_K0(a, b, ell, var))(points_2))(points_1)
    result_matrix = jnp.transpose(blocks, (0,2,1,3)).reshape(n_1*3, n_2*3)
    sigma_noise = 0.05
    if noise and n_1 == n_2:
        result_matrix = result_matrix + sigma_noise**2 * jnp.eye(n_1*3) #terms on the diagonals
    return result_matrix

def fit_hyperparams(train_coords, train_vels, n_restarts=4, jitter=1e-6, seed=0):
    X = jnp.asarray(train_coords)
    y = jnp.asarray(train_vels).reshape(-1, 1)
    n = X.shape[0]
    span = float(np.mean(np.ptp(train_coords, axis=0)))
    yvar = float(np.var(train_vels))

    def nll(log_theta):
        th = jnp.exp(log_theta)
        ell, var, noise = th[0:3], th[3], th[4]
        blocks = jax.vmap(lambda a: jax.vmap(lambda b: Hemholtz_K0(a, b, ell, var))(X))(X)
        K = jnp.transpose(blocks, (0,2,1,3)).reshape(3*n, 3*n) + (noise**2 + jitter) * jnp.eye(3*n)
        c, low = cho_factor(K)
        alpha = cho_solve((c, low), y)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(c)))
        base = 0.5*(y.T @ alpha)[0,0] + 0.5*logdet + 0.5*(3*n)*jnp.log(2*jnp.pi)
        pen = jnp.sum(jnp.maximum(0.0, log_theta[0:3] - jnp.log(3*span))**2)
        return base + 10.0*pen

    rng = np.random.default_rng(seed)
    best = None
    for k in range(n_restarts):
        if k == 0:
            t0 = jnp.log(jnp.array([span/2, span/2, span/2, yvar, 0.05*np.sqrt(yvar)+1e-3]))
        else:
            lo = np.log([span*0.1]*3 + [yvar*0.1, 1e-3])
            hi = np.log([span*2]*3 + [yvar*5, 0.2])
            t0 = jnp.asarray(rng.uniform(lo, hi))
        res = jax.scipy.optimize.minimize(nll, t0, method="BFGS")
        f = float(res.fun)
        if np.isfinite(f) and (best is None or f < best[0]):
            best = (f, np.array(jnp.exp(res.x)), bool(res.success))
    f, t, ok = best
    return {'ell': t[0:3], 'var': float(t[3]), 'noise': float(t[4]), 'nll': f, 'success': ok}

stl_filepath = 'inputs/sphere.stl' #TODO this has to be updated to cylinder.stl of correct dimensions and position!!!
#ground truth values or real world measurements; 'training' for GPR
training_point_n = 100
#TODO below, Field.csv comes from CFD. wall.csv should be replaced with points taken from the loaded stl file though!!! The starting point should become some stl.
ground_truth, bounds = sample('inputs/Field.csv', 'inputs/wall.csv', method='random', num_samples=training_point_n)
training_coords = ground_truth[['x-target', 'y-target', 'z-target']].to_numpy() #leave unflattened for functionality
training_vels = ground_truth[['x-velocity','y-velocity','z-velocity']].to_numpy().reshape(-1,1) #flatten to match dims in equation 1.7

#test points to points at which we seek GPR to evaluate the field
res = 20
x,y,z = np.meshgrid(np.linspace(bounds[0,0], bounds[0,1], res), 
                        np.linspace(bounds[1,0], bounds[1,1], res), 
                        np.linspace(bounds[2,0], bounds[2,1], res),
                        indexing='ij')

test_points = np.stack([x,y,z], axis=-1).reshape(-1,3)
test_point_n = test_points.shape[0]
print(f"Total number of training points: {training_point_n}\nTotal number of test points: {test_point_n}")

tick = time.thread_time()
V_inf = np.array([10, 0, 0]) #TODO this has to be updated to the inlet conditions of the cfd!!!
potential_flow_solver = PotentialFlowSolver(V_inf, stl_filepath)

print(test_points.shape)
potential_flow_field = potential_flow_solver.generate_flow_field(x, y, z)  # keep as (N, 3), drop the .reshape(-1,1)
means_tests = potential_flow_field  # already (N, 3) matching test_points

means_training = scipy.interpolate.griddata(
    points=test_points,           # (N, 3)
    values=potential_flow_field,  # (N, 3) — must match first dim of points
    xi=training_coords,           # (M, 3)
    method='linear'
)  # returns (M, 3)

means_training = 0
tock = time.thread_time()
print(f'Prior means potential field calculated in: {tock-tick:.3f}s')

tick = time.thread_time()
fit = fit_hyperparams(training_coords, training_vels - means_training)
ell, var = jnp.asarray(fit['ell']), fit['var']
tock = time.thread_time()
print(f'Hyperparameter fitting complete in {tock-tick:.3f}s')
print(fit)

tick = time.thread_time()
K_matrix = assemble_dat_shi(training_coords, training_coords, ell, var)
tock = time.thread_time()
print(f'K_matrix assembled in {tock-tick:.3f}s')
print(K_matrix.shape)

#invert the noised K matrix
tick = time.thread_time()
K_noised_inv = jnp.linalg.inv(K_matrix)
tock = time.thread_time()
print(f'Inversion complete in {tock-tick:.3f}s')
print(K_noised_inv.shape)

tick = time.thread_time()
k_star = assemble_dat_shi(test_points, training_coords, ell, var, noise=False)
tock = time.thread_time()
print(f'K_star assembled in {tock-tick:.3f}s')
print(k_star.shape)

print((k_star @ K_noised_inv).shape)
print((training_vels - means_training).shape)

tick = time.thread_time()
residuals = (training_vels.reshape(-1, 3) - means_training).reshape(-1, 1)
GPR_posterior = means_tests.reshape(-1, 1) + k_star @ K_noised_inv @ residuals
tock = time.thread_time()
print(f'GPR Posterior generated in {tock-tick:.3f}s')
print(GPR_posterior.shape)

GPR_posterior_reshaped = np.array(GPR_posterior).reshape(-1,3)


plt.show()

#optional slice visualization
U = GPR_posterior_reshaped.reshape(res, res, res, 3)
P = test_points.reshape(res, res, res, 3)
k = res // 2
Xs, Ys = P[:, :, k, 0], P[:, :, k, 1]
us, vs, ws = U[:, :, k, 0], U[:, :, k, 1], U[:, :, k, 2]
mag = np.sqrt(us**2 + vs**2 + ws**2)
plt.figure(figsize=(7,6))
pc = plt.contourf(Xs, Ys, mag, levels=30, cmap='viridis')
plt.colorbar(pc, label='|velocity|')
plt.quiver(Xs, Ys, us, vs, color='white')
plt.gca().set_aspect('equal')
plt.title(f'xy slice at z={np.linspace(bounds[2,0],bounds[2,1],res)[k]:.2f}')
plt.show()

#TODO add in a GPR for the pressure field
#TODO momentum integral code to calculate forces on the building
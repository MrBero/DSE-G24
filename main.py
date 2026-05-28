import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
import jax.scipy.optimize
import matplotlib.pyplot as plt
from sampling import sample
import trimesh
import time

jax.config.update("jax_enable_x64", True)

#ground truth values or real world measurements; 'training' for GPR
training_point_n = 100
ground_truth, bounds, wall_df = sample('inputs/Field.csv', 'inputs/wall.csv', method='random', num_samples=training_point_n)
training_coords = ground_truth[['x-target', 'y-target', 'z-target']].to_numpy() #leave unflattened for functionality
training_vels = ground_truth[['x-velocity','y-velocity','z-velocity']].to_numpy().reshape(-1,1) #flatten to match dims in equation 1.7
# print(training_vels.shape)


#test points to points at which we seek GPR to evaluate the field
res = 60
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

# def matern52_np(v1, v2, ell_1=1.0, ell_2=2.0, ell_3=1.5, var=1.0):
#     diff = v2 - v1
#     r = np.sqrt((diff[0]/ell_1)**2 + (diff[1]/ell_2)**2 + (diff[2]/ell_3)**2)
#     return var * (1 + np.sqrt(5)*r + (5/3)*r**2) * np.exp(-np.sqrt(5)*r)

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


def gammas_VPM(centroids, normals, V_inf): #vortex panel method
    n = centroids.shape[0]
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                A[i,j] = 0.5
                continue
            r_vect = centroids[i] - centroids[j]
            r = np.linalg.norm(r_vect)
            A[i,j] = np.dot(r_vect / (4 * np.pi * r**3), normals[i])
    RHS = -np.dot(normals, V_inf)
    gammas = np.linalg.solve(A, RHS)
    return gammas


def prior_mean_velocity(P, centers, gammas, V_inf):
    P = np.atleast_2d(P)
    vel = np.tile(np.asarray(V_inf, float), (P.shape[0], 1))
    diff = P[:, None, :] - centers[None, :, :]
    R = np.maximum(np.linalg.norm(diff, axis=2), 1e-12)
    coeff = gammas[None, :] / (4*np.pi * R**3)
    vel += np.einsum('qc,qcd->qd', coeff, diff)
    return vel


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

mesh = trimesh.load_mesh('inputs/sphere.stl')
V_inf = np.array([10, 0, 0])
gammas = gammas_VPM(mesh.triangles_center, mesh.face_normals, V_inf=V_inf)

means_training = prior_mean_velocity(training_coords, mesh.triangles_center, gammas, V_inf).reshape(-1, 1)
means_tests = prior_mean_velocity(test_points, mesh.triangles_center, gammas, V_inf).reshape(-1, 1)

fit = fit_hyperparams(training_coords, training_vels - means_training)
ell, var = jnp.asarray(fit['ell']), fit['var']
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
GPR_posterior = means_tests + k_star @ K_noised_inv @ (training_vels - means_training)
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
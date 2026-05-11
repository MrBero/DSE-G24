#imports
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import jax
import jax.numpy as jnp
from jax import hessian
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


#use x64
jax.config.update("jax_enable_x64", True)

# define kernel 

def matern52(x, xp, var, l):
    dist_sq = jnp.sum((x - xp) ** 2)
    r = jnp.sqrt(dist_sq + 1e-12) # epsilon for stability to avoid 0/0
    s = jnp.sqrt(5.0) * r / l
    return var * (1 + s + s**2 / 3) * jnp.exp(-s) 



# We do helmholtz decomposition as shown in the GPjax guide source. We drop the irrotational component since we want it to be purely incompressible.
def kern_entry(X, Xp, var_psi, l_psi):
    x, z   = X[:2], jnp.int32(X[2]) #We assign incices to split between u (0) and v (1)
    xp, zp = Xp[:2], jnp.int32(Xp[2])

    H_psi = -hessian(matern52, argnums=0)(x, xp, var_psi, l_psi) 
    
    return (-1.0) ** (z + zp) * H_psi[1-z, 1-zp] #flip indices with 1-zp such that for example when we have u and we want v then it becomes 1-0 so 1. 


# build convariance matrix out of single entries based on kernel
_row = jax.vmap(kern_entry, in_axes=(None, 0, None, None))
kern = jax.jit(jax.vmap(_row, in_axes=(0, None, None, None)))

# now apply GP like the source shows (We do not take ownership for these 2 functions, but to our knowledge we're just applying GP)

def predict(X_tr, y_tr, X_te, theta):
    vs, ls, noise = jnp.exp(theta)
    K   = kern(X_tr, X_tr, vs, ls) + noise * jnp.eye(len(y_tr))
    Ks  = kern(X_te, X_tr, vs, ls)
    Kss = kern(X_te, X_te, vs, ls)
    L   = jnp.linalg.cholesky(K)
    a   = jax.scipy.linalg.cho_solve((L, True), y_tr)
    V   = jax.scipy.linalg.solve_triangular(L, Ks.T, lower=True)
    return Ks @ a, jnp.diag(Kss) - jnp.sum(V**2, axis=0)

# optimizing
def neg_log_ml(theta, X_tr, y_tr):
    vs, ls, noise = jnp.exp(theta)
    n = len(y_tr)
    K = kern(X_tr, X_tr, vs, ls) + noise * jnp.eye(n)
    L = jnp.linalg.cholesky(K)
    a = jax.scipy.linalg.cho_solve((L, True), y_tr)
    return 0.5 * (y_tr @ a + 2 * jnp.sum(jnp.log(jnp.diag(L))) + n * jnp.log(2 * jnp.pi))

# end paste

# dummy variable to distinguish u and v 
def to_3d(xy, uv=None):
    N = xy.shape[0]
    z = jnp.tile(jnp.array([0.0, 1.0]), N).reshape(-1, 1)
    X = jnp.hstack([jnp.repeat(xy, 2, axis=0), z])
    if uv is not None:
        y = uv.reshape(-1, order="C")
        return X, y
    return X

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # 1. Load data
    def true_eddy(xy):
        x, y = xy[:, 0], xy[:, 1]
        return np.stack([np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)], axis=1)

    xy_train = rng.random((20, 2)) * 5 
    uv_train = true_eddy(xy_train) + 0.05 * rng.standard_normal((20, 2))

    # 2. Prepare data
    scaler = StandardScaler().fit(xy_train) #scale
    X_tr, y_tr = to_3d(jnp.array(scaler.transform(xy_train)), jnp.array(uv_train)) #separate u and v 

    # 3. Optimize the 3 remaining Hyperparameters
    # Initial guess: log(var)=0, log(lengthscale)=0, log(noise)=-4
    theta_init = np.array([0.0, 0.0, -4.0]) 
    
    print("Optimizing Pure Incompressible GP...")
    res = minimize(
        lambda t: float(neg_log_ml(jnp.array(t), X_tr, y_tr)), 
        theta_init, 
        method="L-BFGS-B"
    )
    print(f"Converged: {res.success} | NLL: {res.fun:.2f}")

    # 4. Predict on a grid to visualize
    x_grid = np.linspace(0, 5, 20)
    y_grid = np.linspace(0, 5, 20)
    xx, yy = np.meshgrid(x_grid, y_grid)
    xy_test = np.vstack([xx.ravel(), yy.ravel()]).T

    X_te = to_3d(jnp.array(scaler.transform(xy_test))) # Note: No 'uv' needed here
    
    mean_pred, var_pred = predict(X_tr, y_tr, X_te, jnp.array(res.x))
    
    # Reshape predictions back to 2D vector field
    uv_pred = np.array(mean_pred).reshape(-1, 2, order="C")
    u_pred = uv_pred[:, 0].reshape(20, 20)
    v_pred = uv_pred[:, 1].reshape(20, 20)

    # 5. Plot the swirl
    plt.figure(figsize=(8, 6))
    plt.title("Pure Incompressible Flow Prediction")
    plt.quiver(xx, yy, u_pred, v_pred, color='blue', alpha=0.6, label='GP Prediction')
    plt.scatter(xy_train[:, 0], xy_train[:, 1], c='red', s=15, label='Sensors (Training Data)')
    plt.legend()
    plt.show()

    # Reshape variance
# Since var_pred comes from the stacked X_te, we split it back to u and v
vars_reshaped = np.array(var_pred).reshape(-1, 2, order="C")
u_var = vars_reshaped[:, 0].reshape(20, 20)
v_var = vars_reshaped[:, 1].reshape(20, 20)

# Total variance (magnitude of uncertainty)
total_var = np.sqrt(u_var**2 + v_var**2)

plt.figure(figsize=(8, 6))
# Plot the heatmap of uncertainty
plt.contourf(xx, yy, total_var, levels=20, cmap='viridis')
plt.colorbar(label='Predictive Uncertainty (Variance)')

# Overlay the training points to see how uncertainty drops near them
plt.scatter(xy_train[:, 0], xy_train[:, 1], c='red', edgecolors='white', label='Sensors')
plt.title("GP Uncertainty Map")
plt.legend()
plt.show()
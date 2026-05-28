import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# GPR Kernels
def matern_five_two_2D(x_1, x_2, y_1, y_2, ell_1, ell_2, var):
    r_ARD = np.sqrt(((x_1 - y_1)**2 / ell_1**2) + ((x_2 - y_2)**2 / ell_2**2))
    term_1 = 1 + np.sqrt(5)*r_ARD + (5/3)*(r_ARD**2)
    term_2 = np.exp(-np.sqrt(5)*r_ARD)
    val = var * term_1 * term_2
    return val

seed = 7
rnd.seed(seed)

# --- 1. Training Data Setup ---
train_count = 10
sigma_measure = 0.05
x_train = rnd.uniform(-10, 10, train_count)
y_train = rnd.uniform(-10, 10, train_count)
z_train = np.sin(x_train) * np.cos(y_train) + rnd.normal(0, sigma_measure, train_count)

# Hyperparameters
ell_1, ell_2, var, noise = 2.0, 2.0, 1.0, sigma_measure**2 # Increased lengthscales slightly for a smoother fit

# --- 2. GPR Calculations ---
# Training matrix
X1, X2 = np.meshgrid(x_train, x_train, indexing='ij')
Y1, Y2 = np.meshgrid(y_train, y_train, indexing='ij')
K_xx = matern_five_two_2D(X1, X2, Y1, Y2, ell_1, ell_2, var) + noise * np.eye(train_count)
K_xx_inv = np.linalg.inv(K_xx)

# Dense 2D Test Grid (This creates the required 2D structure for plot_surface)
test_res = 50
x_space = np.linspace(-10, 10, test_res)
y_space = np.linspace(-10, 10, test_res)
X_mesh, Y_mesh = np.meshgrid(x_space, y_space, indexing='ij') # These are 2D arrays (50x50)

# Flatten for kernel processing
x_test_flat = X_mesh.flatten()
y_test_flat = Y_mesh.flatten()

# Cross-covariance matrix
A_star, B_star = np.meshgrid(x_test_flat, x_train, indexing='ij')
C_star, D_star = np.meshgrid(y_test_flat, y_train, indexing='ij')
K_xstar_x = matern_five_two_2D(A_star, B_star, C_star, D_star, ell_1, ell_2, var)

# Compute mean prediction and reshape it BACK into a 2D matrix
z_pred_flat = K_xstar_x @ K_xx_inv @ z_train
Z_pred_mesh = z_pred_flat.reshape(test_res, test_res) # Now Z is 2D!

# --- 3. Plotting ---
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 8))

# FIX: Pass the 2D mesh grids to plot_surface
surf = ax.plot_surface(X_mesh, Y_mesh, Z_pred_mesh, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.7)

# Scatter plot the 1D training vectors as discrete individual data points
ax.scatter(x_train, y_train, z_train, color='black', s=50, depthshade=False, label="Training Points")

# Customize the axes
ax.set_zlim(-2, 2)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.02f}')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
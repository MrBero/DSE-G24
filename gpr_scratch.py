import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import scipy as sp
import jax as jax

# GPR Kernels
def matern_five_two_1D(a, b, ell, var):
    dist = np.abs(a - b)
    term_1 = 1 + (np.sqrt(5) * dist) / ell + (5 * (dist**2)) / (3 * (ell**2))
    term_2 = (-(np.sqrt(5) * dist)) / ell
    term_3 = np.exp(term_2)
    val = var * term_1 * term_3
    return val

def matern_five_two_2D(x_1, x_2, y_1, y_2, ell_1, ell_2, var):
    r_ARD = np.sqrt(((x_1 - y_1)**2 / ell_1**2) + ((x_2 - y_2)**2 / ell_2**2))
    term_1 = 1 + np.sqrt(5)*r_ARD + (5/3)*(r_ARD**2)
    term_2 = np.exp(-np.sqrt(5)*r_ARD)
    val = var * term_1 * term_2
    return val

seed = 7
rnd.seed(seed)

# Random sampling function
# Inputs: 
train_count = 100
x_train = np.sort(rnd.uniform(-10, 10, train_count))
x_train = np.array(x_train)
y_train = 2 * np.sin(x_train) + rnd.normal(0, 0.5, train_count)
y_train = np.array(y_train)

ell, var, noise = 1.0, 1.0, 0.25
x_test = np.linspace(-10, 10, 1000)
y_ground = 2 * np.sin(x_test)

# Create Covariance Matrix between Training Points
A, B, = np.meshgrid(x_train, x_train, indexing='ij')
K_1D = matern_five_two_1D(A, B, ell, var) + noise * np.eye(train_count)
K_1D_inv = np.linalg.inv(K_1D)

# Create Covariance Vector between Test and Training Points
A_star, B_star = np.meshgrid(x_test, x_train, indexing='ij')
K_xstar = matern_five_two_1D(A_star, B_star, ell, var)

# Covariance between test points themeselves
A_test, B_test = np.meshgrid(x_test, x_test, indexing='ij')
K_xstar_xstar = matern_five_two_1D(A_test, B_test, ell, var)


# GPR Prediction
mu_test = K_xstar @ K_1D_inv @ y_train
Sigma_test = K_xstar_xstar - (K_xstar @ K_1D_inv @ K_xstar.T)
var_test = np.diag(Sigma_test)
std_test = np.sqrt(np.maximum(1e-8, var_test))

print(K_1D)
print(K_1D_inv)
print("Is Covariance Matrix symmetrical? ", np.allclose(K_1D, K_1D.T))
print("Is Covariance Matrix symmetrical? ", np.allclose(K_1D_inv, K_1D_inv.T))

plt.figure(figsize=(16, 10))
plt.plot(x_test, mu_test, 'r-', label='Predictive Mean', lw=2)
plt.plot(x_test, y_ground,'g-', label='Ground Truth', lw=2)
plt.fill_between(x_test, mu_test - 2*std_test, mu_test + 2*std_test, color='red', alpha=0.2, label='95% Confidence Interval')
plt.scatter(x_train, y_train, color='black', marker='x', s=100, label='Training Data')
plt.title('1D Gaussian Process Regression (Matern 5/2)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
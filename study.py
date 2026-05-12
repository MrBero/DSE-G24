import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# --- 1. Load and Clean Data ---
FILE_PATH = 'output.xlsx'
df = pd.read_excel(FILE_PATH)
# Strip whitespace and lowercase to handle 'x-coordir' vs 'x-coordinate'
df.columns = df.columns.str.strip().str.lower()

# Identify Inlet (min X) and Outlet (max X)
x_vals = sorted(df['x-coordinate'].unique())
inlet_df = df[df['x-coordinate'] == x_vals[0]].sort_values('y-coordinate')
outlet_df = df[df['x-coordinate'] == x_vals[-1]].sort_values('y-coordinate')

# Constants
RHO = 1.225 
POINT_RANGE = np.array([5, 10, 15, 20, 25, 30, 50, 60, 80, 100])

def calc_flux(y, u, p):
    """Calculates P + rho*u^2 integrated over y"""
    return np.trapz(p + RHO * (u**2), y)

# Reference "Reality" Drag
y_ref = outlet_df['y-coordinate'].values
u_ref = outlet_df['x-velocity'].values
p_ref = outlet_df['pressure'].values
drag_reality = calc_flux(inlet_df['y-coordinate'], inlet_df['x-velocity'], inlet_df['pressure']) - \
               calc_flux(y_ref, u_ref, p_ref)

# --- 2. GPR vs. Trapezoidal Study ---
gpr_errors = []
trapz_errors = []

# Define RBF Kernel (Constant * RBF)
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

for n in POINT_RANGE:
    # Subsample n points from the outlet
    indices = np.linspace(0, len(y_ref) - 1, n, dtype=int)
    y_train = y_ref[indices].reshape(-1, 1)
    u_train = u_ref[indices]
    p_train = p_ref[indices]
    
    # Method A: Direct Trapezoidal (on n points)
    drag_trapz = calc_flux(inlet_df['y-coordinate'], inlet_df['x-velocity'], inlet_df['pressure']) - \
                 np.trapz(p_train + RHO * (u_train**2), y_train.flatten())
    trapz_errors.append(abs(drag_trapz - drag_reality))
    
    # Method B: GPR with RBF Reconstruction
    gp_u = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20).fit(y_train, u_train)
    gp_p = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20).fit(y_train, p_train)
    
    # Predict back onto high-resolution grid
    u_pred = gp_u.predict(y_ref.reshape(-1, 1))
    p_pred = gp_p.predict(y_ref.reshape(-1, 1))
    
    drag_gpr = calc_flux(inlet_df['y-coordinate'], inlet_df['x-velocity'], inlet_df['pressure']) - \
               calc_flux(y_ref, u_pred, p_pred)
    gpr_errors.append(abs(drag_gpr - drag_reality))

# --- 3. Plotting Results ---
plt.figure(figsize=(10, 6))
plt.plot(POINT_RANGE, trapz_errors, 'o--', label='Trapezoidal Method', color='red', alpha=0.6)
plt.plot(POINT_RANGE, gpr_errors, 's-', label='GPR (RBF Kernel)', color='blue')
plt.yscale('log')
plt.xlabel('Number of Sampling Points ($n$)', fontsize=12)
plt.ylabel('Abs Error in Drag (N/m) [Log Scale]', fontsize=12)
plt.title('Error Convergence: GPR vs. Trapezoidal Integration', fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.show()
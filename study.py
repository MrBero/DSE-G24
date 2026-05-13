import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator, interp1d

# --- 1. Load and Clean Data ---
FILE_PATH = 'output.xlsx'
df = pd.read_excel(FILE_PATH)
df.columns = df.columns.str.strip().str.lower()

x_vals = sorted(df['x-coordinate'].unique())
inlet_df  = df[df['x-coordinate'] == x_vals[0]].sort_values('y-coordinate')
outlet_df = df[df['x-coordinate'] == x_vals[-1]].sort_values('y-coordinate')

RHO = 1.225
POINT_RANGE = np.array([5, 10, 15, 20, 25, 30, 50, 60, 80, 100])

def calc_flux(y, u, p):
    """Momentum flux: integral of (P + rho*u^2) over y."""
    return np.trapz(p + RHO * u**2, y)

# Reference ("truth") values at the outlet
y_ref = outlet_df['y-coordinate'].values
u_ref = outlet_df['x-velocity'].values
p_ref = outlet_df['pressure'].values

# Inlet flux is fixed across the sweep — compute once
inlet_flux = calc_flux(inlet_df['y-coordinate'].values,
                       inlet_df['x-velocity'].values,
                       inlet_df['pressure'].values)
drag_reality = inlet_flux - calc_flux(y_ref, u_ref, p_ref)

# --- 2. Reconstruction methods ---
rbf_kernel    = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
matern_kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0,
                                             length_scale_bounds=(1e-2, 1e2), nu=2.5)

def recon_linear(yt, ft, yq):
    return interp1d(yt, ft, kind='linear', fill_value='extrapolate')(yq)

def recon_cubic_spline(yt, ft, yq):
    return CubicSpline(yt, ft, extrapolate=True)(yq)

def recon_pchip(yt, ft, yq):
    return PchipInterpolator(yt, ft, extrapolate=True)(yq)

def recon_akima(yt, ft, yq):
    return Akima1DInterpolator(yt, ft)(yq)

def recon_gpr(yt, ft, yq, kernel):
    gp = GaussianProcessRegressor(kernel=kernel,
                                  n_restarts_optimizer=10,
                                  normalize_y=True)  # important for pressure scale
    gp.fit(yt.reshape(-1, 1), ft)
    return gp.predict(yq.reshape(-1, 1))

methods = {
    'Trapezoidal (no recon)': 'trapz',
    'Linear':           lambda yt, ft, yq: recon_linear(yt, ft, yq),
    'Cubic Spline':     lambda yt, ft, yq: recon_cubic_spline(yt, ft, yq),
    'PCHIP':            lambda yt, ft, yq: recon_pchip(yt, ft, yq),
    'Akima':            lambda yt, ft, yq: recon_akima(yt, ft, yq),
    'GPR (RBF)':        lambda yt, ft, yq: recon_gpr(yt, ft, yq, rbf_kernel),
    'GPR (Matern 5/2)': lambda yt, ft, yq: recon_gpr(yt, ft, yq, matern_kernel),
}

# --- 3. Sweep over sample counts ---
errors = {name: [] for name in methods}

for n in POINT_RANGE:
    idx = np.linspace(0, len(y_ref) - 1, n, dtype=int)
    y_tr, u_tr, p_tr = y_ref[idx], u_ref[idx], p_ref[idx]

    for name, method in methods.items():
        if method == 'trapz':
            # Integrate directly on the sparse grid — no reconstruction
            outlet_flux = np.trapz(p_tr + RHO * u_tr**2, y_tr)
        else:
            u_pred = method(y_tr, u_tr, y_ref)
            p_pred = method(y_tr, p_tr, y_ref)
            outlet_flux = calc_flux(y_ref, u_pred, p_pred)

        errors[name].append(abs((inlet_flux - outlet_flux) - drag_reality))

# --- 4. Plot ---
styles = {
    'Trapezoidal (no recon)': ('o--', 'red'),
    'Linear':                 ('v-',  'orange'),
    'Cubic Spline':           ('^-',  'green'),
    'PCHIP':                  ('D-',  'purple'),
    'Akima':                  ('P-',  'brown'),
    'GPR (RBF)':              ('s-',  'royalblue'),
    'GPR (Matern 5/2)':       ('*-',  'navy'),
}

plt.figure(figsize=(11, 7))
for name, errs in errors.items():
    marker, color = styles[name]
    plt.plot(POINT_RANGE, errs, marker, label=name, color=color, alpha=0.85, markersize=7)

plt.yscale('log')
plt.xlabel('Number of Sampling Points ($n$)', fontsize=12)
plt.ylabel('Abs Error in Drag (N/m) [Log Scale]', fontsize=12)
plt.title('Drag Reconstruction Error vs. Sampling Density', fontsize=14)
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend(loc='best', frameon=True)
plt.tight_layout()
plt.show()
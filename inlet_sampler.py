import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

# =========================================================================== #
#  CONFIG                                                                      #
# =========================================================================== #
U_mean     = 12.0   # mean inlet wind speed          [m/s]
I_u        = 0.1   # turbulence intensity  σ_u/U    [-]
T_int      = 5.0    # integral time scale             [s]
fs         = 10.0   # drone sampling rate             [Hz]
T_total    = 600  # total simulation time           [s]

epsilon_ci = 0.05   # Cond 1: Z·σ_Ū/Ū  < 5 %        [-]
delta_stab = 0.01   # Cond 2: mean drift < 1 %        [-]
Z_score    = 1.96   # 95 % confidence
N_eff_min  = 10     # Cond 3: min independent gusts

ema_alpha  = 0.05   # EMA smoothing on T_int_est  (smaller = smoother)
seed       = 4
# =========================================================================== #

sigma_u   = I_u * U_mean
dt        = 1.0 / fs
N         = int(T_total * fs)
phi       = np.exp(-dt / T_int)
sigma_eps = sigma_u * np.sqrt(1 - phi**2)

# Fixed burn-in: must collect at least this many seconds before any condition
# is evaluated.  Use the prior T_int, NOT the online estimate.
BURNIN_T  = 5.0 * T_int          # [s]
BURNIN_N  = int(BURNIN_T * fs)   # [samples]

T_min_theory = Z_score**2 * 2 * I_u**2 * T_int / epsilon_ci**2

print("=" * 60)
print("  LAWSS INLET SAMPLER v2 — CONFIGURATION")
print("=" * 60)
print(f"  U_mean  = {U_mean} m/s   I_u = {I_u}   σ_u = {sigma_u:.3f} m/s")
print(f"  T_int   = {T_int} s      fs  = {fs} Hz")
print(f"  ε_ci    = {epsilon_ci*100:.0f}%   δ_stab = {delta_stab*100:.0f}%   Z = {Z_score}")
print(f"  N_eff_min = {N_eff_min}   burn-in = {BURNIN_T:.0f} s")
print(f"  Theoretical T_min = {T_min_theory:.1f} s")
print("=" * 60)

# =========================================================================== #
#  STEP 1 — AR(1) turbulence signal                                            #
# =========================================================================== #
rng     = np.random.default_rng(seed)
u_prime = np.zeros(N)
eps     = rng.normal(0.0, sigma_eps, N)
for n in range(1, N):
    u_prime[n] = phi * u_prime[n-1] + eps[n]
velocity = U_mean + u_prime
time_arr = np.arange(N) * dt

# =========================================================================== #
#  STEP 2–5 — Online estimation loop                                           #
# =========================================================================== #
run_mean_arr  = np.full(N, np.nan)
T_int_est_arr = np.full(N, np.nan)
ci_rel_arr    = np.full(N, np.nan)
stab_arr      = np.full(N, np.nan)
Neff_arr      = np.full(N, np.nan)

# --- Welford: grand mean & variance ---
wf_n = 0; wf_mean = 0.0; wf_M2 = 0.0

# --- Lag-1 estimator (FIX 1): dedicated pair-mean accumulator ---
# We track E[u_k], E[u_{k-1}], E[u_k * u_{k-1}], E[u_k^2] over lag pairs.
# Using the unbiased online formula:
#   cov  = mean(u_k * u_{k-1}) - mean(u_k) * mean(u_{k-1})
#   var  = mean(u_k^2)         - mean(u_k)^2
# Because the series is stationary, mean(u_k) ≈ mean(u_{k-1}) ≈ μ,
# so we share one mean accumulator for both.
L_n   = 0        # number of lag-1 pairs seen
L_mux = 0.0      # running mean of u[k]   (the "current" sample in each pair)
L_muy = 0.0      # running mean of u[k-1] (the "lagged"  sample in each pair)
L_mxy = 0.0      # running mean of u[k]*u[k-1]
L_mx2 = 0.0      # running mean of u[k]^2

T_int_est = T_int     # prior — used until burn-in completes

# --- Stability buffer: fixed length = 5*T_int ---
STAB_WIN = int(5.0 * T_int * fs)
mean_hist = np.full(STAB_WIN, np.nan)
hist_ptr  = 0

stop_idx = None; stop_time = None

for n in range(N):
    u    = velocity[n]
    prev = velocity[n-1] if n > 0 else u

    # ---- Welford ----
    wf_n   += 1
    d       = u - wf_mean
    wf_mean += d / wf_n
    wf_M2  += d * (u - wf_mean)
    run_mean_arr[n] = wf_mean

    # ---- Lag-1 accumulator (FIX 1) ----
    if n > 0:
        L_n   += 1
        # Online mean updates (Welford-style for each quantity)
        L_mux += (u      - L_mux) / L_n
        L_muy += (prev   - L_muy) / L_n
        L_mxy += (u*prev - L_mxy) / L_n
        L_mx2 += (u*u    - L_mx2) / L_n

        if L_n > 20:
            cov_lag = L_mxy - L_mux * L_muy
            var_lag = max(L_mx2 - L_mux**2, 1e-12)
            rho1    = np.clip(cov_lag / var_lag, 1e-6, 1 - 1e-6)
            T_new   = -dt / np.log(rho1)
            # FIX 3: EMA smoothing so one bad sample cannot corrupt CI
            T_int_est = ema_alpha * T_new + (1 - ema_alpha) * T_int_est
            T_int_est_arr[n] = T_int_est

    # ---- Stability buffer ----
    mean_hist[hist_ptr % STAB_WIN] = wf_mean
    hist_ptr += 1

    # ---- FIX 2: Fixed burn-in gate ----
    if n < BURNIN_N:
        continue

    current_T = (n + 1) * dt

    # ---- Step 4: standard error of the mean ----
    var_u      = wf_M2 / max(wf_n - 1, 1)
    sigma_Ubar = np.sqrt(max(2.0 * var_u * T_int_est / current_T, 0.0))

    # ---- Step 5: conditions ----
    # Cond 3
    N_eff = current_T / (2.0 * T_int_est)
    cond3 = N_eff >= N_eff_min
    Neff_arr[n] = N_eff

    # Cond 1
    ci_rel = Z_score * sigma_Ubar / max(abs(wf_mean), 1e-9)
    cond1  = ci_rel < epsilon_ci
    ci_rel_arr[n] = ci_rel

    # Cond 2
    oldest = mean_hist[hist_ptr % STAB_WIN]
    if np.isfinite(oldest):
        stab = abs(wf_mean - oldest) / max(abs(wf_mean), 1e-9)
    else:
        stab = np.inf
    cond2 = stab < delta_stab
    stab_arr[n] = stab

    if cond1 and cond2 and cond3 and stop_idx is None:
        stop_idx  = n
        stop_time = current_T

# =========================================================================== #
#  STEP 6 — Report                                                             #
# =========================================================================== #
print("\n  RESULTS")
print("=" * 60)
if stop_time:
    est = run_mean_arr[stop_idx]
    err = abs(est - U_mean) / U_mean * 100
    print(f"  Stopped at        : {stop_time:.1f} s")
    print(f"  Theoretical T_min : {T_min_theory:.1f} s")
    print(f"  Estimated mean    : {est:.4f} m/s  (true: {U_mean})")
    print(f"  Relative error    : {err:.2f} %")
    T_est_stop = T_int_est_arr[stop_idx]
    print(f"  T_int estimated   : {T_est_stop:.2f} s  (true: {T_int})")
    print(f"  N_eff at stop     : {Neff_arr[stop_idx]:.1f}")
else:
    print("  *** Conditions not met within simulation window ***")
print("=" * 60)

# =========================================================================== #
#  PLOTTING                                                                    #
# =========================================================================== #
fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True,
                         constrained_layout=True)
fig.suptitle('LAWSS — Inlet Plane Sampling Pipeline  (v2 corrected)',
             fontsize=13, fontweight='bold')

def vline(ax):
    if stop_time:
        ax.axvline(stop_time, color='#a8ff78', ls='--', lw=1.4,
                   label=f'Stop {stop_time:.0f} s')

# Panel 1 — velocity + mean
ax = axes[0]
ax.plot(time_arr, velocity, '#00d4ff', lw=0.4, alpha=0.55, label='u(t)')
ax.plot(time_arr, run_mean_arr, '#ff6b6b', lw=1.8, label='Running mean Ū')
ax.axhline(U_mean, color='white', lw=1.2, ls=':', label=f'True mean {U_mean} m/s')
vline(ax); ax.legend(fontsize=7, ncol=4)
ax.set_ylabel('Velocity  [m/s]')
ax.set_title('Instantaneous velocity and running mean')

# Panel 2 — T_int estimate (EMA-smoothed lag-1)
ax = axes[1]
# Clamp for display so early transients don't crush the scale
ax.plot(time_arr, np.clip(T_int_est_arr, 0, 4*T_int), '#f953c6', lw=1.2,
        label='T_int estimate (EMA-smoothed lag-1)')
ax.axhline(T_int, color='white', lw=1.2, ls=':', label=f'True T_int = {T_int} s')
ax.axvline(BURNIN_T, color='gray', lw=1.0, ls=':', alpha=0.6, label='Burn-in end')
vline(ax); ax.legend(fontsize=7, ncol=3)
ax.set_ylabel('T_int  [s]')
ax.set_ylim(0, 4 * T_int)
ax.set_title('Online T_int estimate (EMA-smoothed lag-1 ACF)')

# Panel 3 — CI + N_eff
ax = axes[2]
ax.plot(time_arr, ci_rel_arr * 100, '#f7971e', lw=1.3,
        label=f'Z·σ_Ū/Ū  (Cond 1, target < {epsilon_ci*100:.0f}%)')
ax.axhline(epsilon_ci * 100, color='yellow', ls='--', lw=1.2,
           label=f'ε = {epsilon_ci*100:.0f}%')
ax_r = ax.twinx()
ax_r.plot(time_arr, Neff_arr, '#43e97b', lw=0.9, alpha=0.7,
          label=f'N_eff (Cond 3, ≥{N_eff_min})')
ax_r.axhline(N_eff_min, color='#43e97b', ls=':', lw=1.0)
ax_r.set_ylabel('N_eff', color='#43e97b')
ax_r.tick_params(axis='y', colors='#43e97b')
vline(ax); ax.set_ylabel('Z·σ_Ū/Ū  [%]')
ax.set_ylim(0, epsilon_ci * 100 * 6)
ax.set_title('Cond 1 (CI) and Cond 3 (independent samples)')
l1, lb1 = ax.get_legend_handles_labels()
l2, lb2 = ax_r.get_legend_handles_labels()
ax.legend(l1+l2, lb1+lb2, fontsize=7, ncol=3)

# Panel 4 — stability
ax = axes[3]
ax.plot(time_arr, stab_arr * 100, '#4facfe', lw=1.2,
        label=f'|ΔŪ|/Ū over 5·T_int window (Cond 2)')
ax.axhline(delta_stab * 100, color='yellow', ls='--', lw=1.2,
           label=f'δ = {delta_stab*100:.0f}%')
vline(ax); ax.legend(fontsize=7, ncol=3)
ax.set_xlabel('Time  [s]')
ax.set_ylabel('Relative drift  [%]')
ax.set_ylim(0, delta_stab * 100 * 10)
ax.set_title('Cond 2 — Running mean stability')

for ax in axes:
    ax.grid(alpha=0.15)

plt.show()
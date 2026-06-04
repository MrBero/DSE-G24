import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

# =========================================================================== #
#  CONFIG — edit to match your structure and site                              #
# =========================================================================== #
U_inf       = 15.0   # free-stream wind speed               [m/s]
U_wake      = 9.0    # mean velocity in the wake            [m/s]  (~0.6·U_inf)
I_u_rand    = 0.22   # random turbulence intensity          [-]
I_u_shed    = 0.15   # periodic shedding intensity          [-]
T_int_rand  = 5.0    # true integral time scale (random)    [s]

D           = 20.0   # characteristic building dimension    [m]
St          = 0.10   # Strouhal number                      [-]
fs          = 10.0   # drone sampling rate                   [Hz]
T_total     = 600.0  # total simulation time                 [s]

epsilon_ci  = 0.07   # Cond 1 threshold (relaxed to 10 %)   [-]
delta_stab  = 0.02   # Cond 2 threshold (relaxed to 2 %)    [-]
Z_score     = 1.645  # 90 % confidence (relaxed from 95 %)
N_eff_min   = 10     # Cond 3 min independent samples
N_shed_min  = 5      # Cond 4 min complete shedding cycles

N_acf_update  = 50   # re-estimate ACF every N samples
acf_ema_alpha = 0.3  # EMA smoothing  (new = α·estimate + (1-α)·old)

seed = 42
# =========================================================================== #


# --------------------------------------------------------------------------- #
#  Derived constants                                                           #
# --------------------------------------------------------------------------- #
dt          = 1.0 / fs
N           = int(T_total * fs)
sigma_rand  = I_u_rand * U_wake          # std dev of random component  [m/s]
sigma_shed  = I_u_shed * U_wake          # std dev of shedding component [m/s]
f_shed      = St * U_inf / D            # vortex shedding frequency     [Hz]
T_shed      = 1.0 / f_shed              # shedding period               [s]

# AR(1) coefficients for random component
phi_rand    = np.exp(-dt / T_int_rand)
sigma_eps   = sigma_rand * np.sqrt(1 - phi_rand**2)

# Buffer size for ACF estimation: cover 4 shedding periods
acf_buf_len = max(int(4 * T_shed * fs), 400)
# Max lag to search for zero-crossing: 1.5 shedding periods
acf_max_lag = max(int(1.5 * T_shed * fs), 150)

# Theoretical effective T_int (what the ACF estimator should recover)
alpha_frac  = sigma_rand**2 / (sigma_rand**2 + sigma_shed**2)  # random fraction
T_int_eff_theory = alpha_frac * T_int_rand

# Theoretical minimum time
T_min_theory = Z_score**2 * 2 * (I_u_rand**2 + I_u_shed**2) * T_int_eff_theory / epsilon_ci**2

print("=" * 65)
print("  LAWSS WAKE SAMPLER (DCMA) — CONFIGURATION")
print("=" * 65)
print(f"  U_inf       = {U_inf:.1f} m/s    U_wake = {U_wake:.1f} m/s")
print(f"  I_u_rand    = {I_u_rand:.3f}  → σ_rand = {sigma_rand:.3f} m/s")
print(f"  I_u_shed    = {I_u_shed:.3f}  → σ_shed = {sigma_shed:.3f} m/s")
print(f"  T_int_rand  = {T_int_rand:.1f} s   (true random integral scale)")
print(f"  f_shed      = {f_shed:.4f} Hz   T_shed = {T_shed:.2f} s")
print(f"  α (random variance fraction) = {alpha_frac:.3f}")
print(f"  T_int_eff   = α·T_int_rand = {T_int_eff_theory:.2f} s  (target for ACF estimator)")
print(f"  lag-1 naive T_int estimate  ≈ {-dt/np.log(np.clip(phi_rand*alpha_frac + (1-alpha_frac)*np.cos(2*np.pi*f_shed*dt),1e-9,1-1e-9)):.1f} s  (WRONG — shows why lag-1 fails)")
print(f"  Theoretical T_min           = {T_min_theory:.1f} s  ({T_min_theory/60:.2f} min)")
print(f"  ε_ci = {epsilon_ci*100:.0f}%   δ_stab = {delta_stab*100:.0f}%   Z = {Z_score:.3f}")
print(f"  N_eff_min = {N_eff_min}   N_shed_min = {N_shed_min}")
print("=" * 65)


# =========================================================================== #
#  STEP 1 — Generate wake velocity signal                                      #
# =========================================================================== #
rng = np.random.default_rng(seed)

# Random AR(1) turbulence
u_rand = np.zeros(N)
eps    = rng.normal(0.0, sigma_eps, N)
for n in range(1, N):
    u_rand[n] = phi_rand * u_rand[n-1] + eps[n]

# Periodic vortex shedding  (random arrival phase)
phase0    = rng.uniform(0, 2 * np.pi)
time_arr  = np.arange(N) * dt
u_shed    = sigma_rand * np.sqrt(2) * np.sin(2 * np.pi * f_shed * time_arr + phase0)
# Note: amplitude chosen so std ≈ sigma_shed

velocity = U_wake + u_rand + u_shed


# =========================================================================== #
#  ACF ZERO-CROSSING INTEGRATOR  (the core innovation for the wake)           #
# =========================================================================== #
def acf_zero_crossing_T_int(buf, dt, max_lag):
    n   = len(buf)
    if n < max_lag + 2:
        return np.nan

    x   = buf - buf.mean()
    var = np.var(x)
    if var < 1e-12:
        return np.nan

    lags   = np.arange(max_lag + 1)
    acf    = np.array([np.mean(x[:n-k] * x[k:]) for k in lags]) / var

    # Find first zero-crossing by sign change
    sign_changes = np.where(np.diff(np.sign(acf)))[0]
    if len(sign_changes) == 0:
        return np.nan

    k0 = sign_changes[0]                   # last positive lag before crossing
    # Linear interpolation to sub-lag precision
    frac    = acf[k0] / (acf[k0] - acf[k0 + 1])
    tau_zc  = (k0 + frac) * dt             # zero-crossing time  [s]

    # Integrate ACF from 0 to τ_zc using trapezoid rule
    tau_grid = lags[:k0 + 1] * dt
    T_est    = float(np.trapz(acf[:k0 + 1], tau_grid))

    return max(T_est, dt)   # physical lower bound


# =========================================================================== #
#  STEP 2–5 — Online estimation loop                                           #
# =========================================================================== #
# Storage arrays (for visualisation only)
run_mean_arr      = np.full(N, np.nan)
run_std_arr       = np.full(N, np.nan)
T_int_eff_arr     = np.full(N, np.nan)
T_int_lag1_arr    = np.full(N, np.nan)   # lag-1 estimate for comparison
ci_rel_arr        = np.full(N, np.nan)
stab_arr          = np.full(N, np.nan)
Neff_arr          = np.full(N, np.nan)
n_cycles_arr      = np.full(N, np.nan)

# Welford accumulators
wf_n    = 0
wf_mean = 0.0
wf_M2   = 0.0

# Lag-1 accumulators (kept for diagnostic comparison only)
lag1_xy = 0.0; lag1_x = 0.0; lag1_x2 = 0.0; lag1_n = 0
prev_u  = velocity[0]

# ACF circular buffer
acf_buffer = np.zeros(acf_buf_len)
buf_idx    = 0

# Current T_int_eff estimate
T_int_eff_cur = T_int_eff_theory  # prior fallback

# Stability buffer
stab_win_samples = int(8 * T_int_eff_theory * fs)
mean_history = np.full(max(stab_win_samples, 1), np.nan)
hist_idx     = 0

stop_idx  = None
stop_time = None

for n in range(N):
    u = velocity[n]

    # ------------------------------------------------------------------ #
    #  STEP 2 — Welford update                                            #
    # ------------------------------------------------------------------ #
    wf_n    += 1
    delta    = u - wf_mean
    wf_mean += delta / wf_n
    wf_M2   += delta * (u - wf_mean)
    run_mean_arr[n] = wf_mean

    mean_history[hist_idx % stab_win_samples] = wf_mean
    hist_idx += 1

    # ------------------------------------------------------------------ #
    #  Lag-1 update (for diagnostic comparison)                           #
    # ------------------------------------------------------------------ #
    if n > 0:
        lag1_xy += u * prev_u
        lag1_x  += u
        lag1_x2 += u * u
        lag1_n  += 1
        if lag1_n > 20:
            mu  = lag1_x / lag1_n
            cov = lag1_xy / lag1_n - mu**2
            var = max(lag1_x2 / lag1_n - mu**2, 1e-12)
            r1  = np.clip(cov / var, 1e-6, 1 - 1e-6)
            T_int_lag1_arr[n] = -dt / np.log(r1)
    prev_u = u

    # ------------------------------------------------------------------ #
    #  ACF circular buffer update + periodic re-estimation               #
    # ------------------------------------------------------------------ #
    acf_buffer[buf_idx % acf_buf_len] = u
    buf_idx += 1

    if n > 0 and n % N_acf_update == 0 and buf_idx >= acf_buf_len:
        # Reconstruct ordered buffer contents
        start    = buf_idx % acf_buf_len
        ordered  = np.concatenate([acf_buffer[start:], acf_buffer[:start]])
        T_new    = acf_zero_crossing_T_int(ordered, dt, acf_max_lag)
        if np.isfinite(T_new) and T_new > 0:
            # Exponential moving average to smooth noisy estimates
            T_int_eff_cur = (acf_ema_alpha * T_new
                             + (1 - acf_ema_alpha) * T_int_eff_cur)
    T_int_eff_arr[n] = T_int_eff_cur

    # Update stability window length dynamically
    stab_win_samples = max(int(8 * T_int_eff_cur * fs), 2)

    # ------------------------------------------------------------------ #
    #  STEP 4 — Standard error of the mean                                #
    # ------------------------------------------------------------------ #
    current_T = (n + 1) * dt
    var_u     = wf_M2 / max(wf_n - 1, 1)
    sigma_Ubar = np.sqrt(max(2 * var_u * T_int_eff_cur / current_T, 0.0))

    # ------------------------------------------------------------------ #
    #  STEP 5 — Evaluate stopping conditions                              #
    # ------------------------------------------------------------------ #
    if current_T < max(5 * T_int_eff_cur, N_shed_min * T_shed):
        continue

    # Cond 4: shedding-cycle flush
    n_cycles = current_T / T_shed
    cond4 = n_cycles >= N_shed_min
    n_cycles_arr[n] = n_cycles

    # Cond 3: effective independent samples
    N_eff = current_T / (2 * T_int_eff_cur)
    cond3 = N_eff >= N_eff_min
    Neff_arr[n] = N_eff

    # Cond 1: CI check
    if abs(wf_mean) > 1e-9:
        ci_rel = Z_score * sigma_Ubar / abs(wf_mean)
    else:
        ci_rel = np.inf
    cond1 = ci_rel < epsilon_ci
    ci_rel_arr[n] = ci_rel

    # Cond 2: stability
    oldest = mean_history[hist_idx % min(stab_win_samples, stab_win_samples)]
    if np.isfinite(oldest) and abs(wf_mean) > 1e-9:
        stab = abs(wf_mean - oldest) / abs(wf_mean)
    else:
        stab = np.inf
    cond2 = stab < delta_stab
    stab_arr[n] = stab

    if cond1 and cond2 and cond3 and cond4 and stop_idx is None:
        stop_idx  = n
        stop_time = current_T


# =========================================================================== #
#  STEP 6 — Report                                                             #
# =========================================================================== #
print("\n  RESULTS")
print("=" * 65)
print(f"  True wake mean (unknown in field) : {U_wake:.3f} m/s")
if stop_time is not None:
    est  = run_mean_arr[stop_idx]
    err  = abs(est - U_wake) / U_wake * 100
    T_eff_est = T_int_eff_arr[stop_idx]
    lag1_est  = T_int_lag1_arr[stop_idx] if np.isfinite(T_int_lag1_arr[stop_idx]) else np.nan
    print(f"  Stopped at             : {stop_time:.1f} s  ({stop_time/60:.2f} min)")
    print(f"  Theoretical T_min      : {T_min_theory:.1f} s  ({T_min_theory/60:.2f} min)")
    print(f"  7-min budget used      : {stop_time/420*100:.0f} %")
    print(f"  Estimated mean         : {est:.3f} m/s")
    print(f"  Relative error         : {err:.2f} %  (target < {epsilon_ci*100:.0f} %)")
    print(f"  T_int_eff (ACF est)    : {T_eff_est:.2f} s  (theory: {T_int_eff_theory:.2f} s)")
    print(f"  T_int (lag-1, WRONG)   : {lag1_est:.1f} s  ← shows why lag-1 fails")
    print(f"  N_eff at stop          : {Neff_arr[stop_idx]:.1f}")
    print(f"  Shedding cycles        : {n_cycles_arr[stop_idx]:.1f}")
else:
    print("  *** CONDITIONS NOT MET within simulation window ***")
print("=" * 65)


# =========================================================================== #
#  PLOTTING                                                                    #
# =========================================================================== #
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True,
                         constrained_layout=True)
fig.suptitle('LAWSS — Wake Plane Sampling Pipeline  (DCMA)',
             fontsize=13, fontweight='bold')

C_SIG  = '#00d4ff'
C_MEAN = '#ff6b6b'
C_TRUE = 'white'
C_STOP = '#a8ff78'
C_ACF  = '#d4fc79'
C_LAG  = '#f953c6'

def vline(ax):
    if stop_time:
        ax.axvline(stop_time, color=C_STOP, ls='--', lw=1.4,
                   label=f'Stop {stop_time:.0f} s')

# Panel 1 — raw velocity + running mean
ax = axes[0]
ax.plot(time_arr, velocity, color=C_SIG, lw=0.35, alpha=0.5, label='u(t)')
ax.plot(time_arr, run_mean_arr, color=C_MEAN, lw=1.8, label='Running mean Ū')
ax.axhline(U_wake, color=C_TRUE, lw=1.2, ls=':', label=f'True mean {U_wake} m/s')
vline(ax)
ax.set_ylabel('Velocity  [m/s]')
ax.set_title('Instantaneous velocity + running mean (wake with periodic shedding)')
ax.legend(fontsize=7, ncol=5)

# Panel 2 — T_int estimates: ACF vs lag-1
ax = axes[1]
ax.plot(time_arr, T_int_eff_arr, color=C_ACF, lw=1.5,
        label='T_int_eff (ACF zero-crossing)  ← CORRECT')
ax.plot(time_arr, np.clip(T_int_lag1_arr, 0, 150), color=C_LAG, lw=1.0,
        alpha=0.8, label='T_int (lag-1)  ← WRONG in wake')
ax.axhline(T_int_eff_theory, color=C_TRUE, lw=1.2, ls=':',
           label=f'True T_int_eff = {T_int_eff_theory:.2f} s')
ax.axhline(T_int_rand, color='cyan', lw=0.8, ls='-.',
           label=f'True T_int_rand = {T_int_rand} s')
vline(ax)
ax.set_ylabel('T_int estimate  [s]')
ax.set_title('ACF zero-crossing vs lag-1 estimator — lag-1 fails due to shedding')
ax.legend(fontsize=7, ncol=3)
ax.set_ylim(0, 100)

# Panel 3 — CI (Cond 1) + N_eff (Cond 3) + n_cycles (Cond 4)
ax = axes[2]
ax.plot(time_arr, ci_rel_arr * 100, color='#f7971e', lw=1.3,
        label=f'Z·σ_Ū/Ū  (Cond 1, target < {epsilon_ci*100:.0f}%)')
ax.axhline(epsilon_ci * 100, color='yellow', ls='--', lw=1.2,
           label=f'ε = {epsilon_ci*100:.0f}%')
ax_r = ax.twinx()
ax_r.plot(time_arr, Neff_arr, color='#43e97b', lw=0.9, alpha=0.8,
          label=f'N_eff (Cond 3, ≥{N_eff_min})')
ax_r.plot(time_arr, n_cycles_arr, color='#4facfe', lw=0.9, alpha=0.8,
          label=f'n_cycles (Cond 4, ≥{N_shed_min})')
ax_r.axhline(N_eff_min, color='#43e97b', ls=':', lw=1.0)
ax_r.axhline(N_shed_min, color='#4facfe', ls=':', lw=1.0)
ax_r.set_ylabel('Count', color='white')
vline(ax)
ax.set_ylabel('Z·σ_Ū/Ū  [%]')
ax.set_title('Cond 1 (CI) + Cond 3 (N_eff) + Cond 4 (shedding cycles)')
ax.set_ylim(0, epsilon_ci * 100 * 8)
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax_r.get_legend_handles_labels()
ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7, ncol=3)

# Panel 4 — stability (Cond 2)
ax = axes[3]
ax.plot(time_arr, stab_arr * 100, color='#4facfe', lw=1.2,
        label=f'|ΔŪ|/Ū over 8·T_int window (Cond 2)')
ax.axhline(delta_stab * 100, color='yellow', ls='--', lw=1.2,
           label=f'δ = {delta_stab*100:.0f}%')
vline(ax)
ax.set_xlabel('Time  [s]')
ax.set_ylabel('Relative drift  [%]')
ax.set_title('Cond 2 — Running mean stability (wake meandering guard)')
ax.set_ylim(0, delta_stab * 100 * 12)
ax.legend(fontsize=7, ncol=3)

for ax in axes:
    ax.grid(alpha=0.15)

plt.savefig('lawss_wake_pipeline.png',
            dpi=600, bbox_inches='tight')
print("\nFigure saved → lawss_wake_pipeline.png")
plt.show()
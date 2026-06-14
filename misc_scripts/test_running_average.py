"""
wake_sampling_algorithm.py
==========================
Adaptive minimum-dwell-time algorithm for DCMA (Downstream Conditions Measurement
Array) drone hover measurements in the turbulent wake of a large-scale structure.

Context: LAWSS (Large-Scale Aerodynamic Wake Survey System), TU Delft DSE Group 24.
This is the wake-adapted version of the inlet sampling algorithm. It addresses
the two fundamental differences between inlet and wake:

  1. HIGH TURBULENCE INTENSITY  – up to ~35–40% vs ~10% at inlet.
     Required averaging time scales as T ∝ Iu² · T_int / ε², so Iu alone
     increases the required time by (0.35/0.10)² = 12× relative to inlet.

  2. VORTEX SHEDDING (PERIODIC COMPONENT) – creates a large quasi-periodic
     velocity oscillation that catastrophically biases the standard lag-1
     autocorrelation T_int estimator. A 20 m building at 15 m/s sheds at
     St·U/D ≈ 0.075 Hz (T_shed ≈ 13 s). The lag-1 estimator sees this as
     a ~90 s integral time scale — a 27× overestimate — which would push
     required averaging time to 35+ minutes. This module fixes that.

ENGINEERING SOLUTION: three-part strategy
  (A) Correct T_int estimator: multi-lag ACF with zero-crossing integration.
      Robust to the periodic shedding component; converges to the true
      broadband random turbulence timescale (~3–5 s).
  (B) Relaxed per-drone thresholds: 90 % CI, ε = 10 % per point.
      GPR spatial smoothing across the swarm recovers system-level < 5 %
      accuracy in the momentum-integral force calculation.
  (C) Shedding-cycle flush (Condition 4): enforce a minimum of N_shed_min
      complete shedding cycles so the periodic bias in the running mean is
      washed out before the algorithm can stop.

RESULT: theoretical minimum dwell time ≈ 1.8–3 min, well within the 7 min
budget (two measurements per 14 min battery cycle).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PHYSICAL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
seed = 42
np.random.seed(seed)

T_total = 540       # [s]  9-minute simulation window
fs      = 10        # [Hz] sampling rate
dt      = 1.0 / fs
N       = int(T_total * fs)
time    = np.arange(N) * dt

# Inlet free-stream (known at ground level via ICMA before DCMA deploys)
U_inlet = 15.0       # [m/s]

# ── Wake mean velocity (velocity-deficit region) ─────────────────────────────
velocity_deficit = 0.35           # 35 % deficit — near-wake, bluff body
U_wake           = U_inlet * (1.0 - velocity_deficit)   # ≈ 9.75 m/s
# (True mean — unknown to the algorithm during flight)

# ── Vortex shedding (Strouhal scaling) ───────────────────────────────────────
D_char   = 20.0                           # [m] characteristic body dimension
St       = 0.10                           # Strouhal number (rectangular bluff body)
f_shed   = St * U_inlet / D_char          # 0.075 Hz
T_shed   = 1.0 / f_shed                   # ≈ 13.3 s

# ── Signal components ─────────────────────────────────────────────────────────
# Background broadband turbulence (shear-layer generated, small-scale eddies)
Iu_random     = 0.20                      # 20 % intensity, broadband
sigma_random  = Iu_random * U_wake        # 1.95 m/s
T_int_random  = 5.0                       # [s] true broadband T_int

# Periodic vortex-shedding component
A_shed = 0.15 * U_wake                    # amplitude ≈ 15 % of local mean

# Composite turbulence intensity (what an anemometer actually measures)
sigma_total = np.sqrt(sigma_random**2 + 0.5 * A_shed**2)
Iu_total    = sigma_total / U_wake        # ≈ 22 %

print("=" * 60)
print("WAKE SIGNAL PARAMETERS")
print("=" * 60)
print(f"  U_inlet = {U_inlet:.1f} m/s  |  Velocity deficit = {velocity_deficit*100:.0f}%")
print(f"  U_wake  = {U_wake:.2f} m/s  (true mean — unknown to algorithm)")
print(f"  D_char  = {D_char:.0f} m,  St = {St},  f_shed = {f_shed:.4f} Hz")
print(f"  T_shed  = {T_shed:.2f} s")
print(f"  sigma_random = {sigma_random:.2f} m/s  (Iu_random = {Iu_random*100:.0f}%)")
print(f"  A_shed       = {A_shed:.2f} m/s  (periodic amplitude)")
print(f"  sigma_total  = {sigma_total:.2f} m/s  (Iu_total = {Iu_total*100:.1f}%)")

# ── Theoretical minimum dwell time comparison ─────────────────────────────────
# Inlet (95% CI, ε=5%, T_int=8s, Iu=10%)
T_req_inlet = (1.96**2) * 2 * (0.10**2) * 8.0 / (0.05**2)

# Wake NAIVE: apply inlet formula with raw Iu_total and T_int = T_shed
# (What happens if lag-1 estimator overestimates T_int as T_shed)
T_req_wake_naive = (1.96**2) * 2 * (Iu_total**2) * T_shed / (0.05**2)

# Wake CORRECTED: relaxed thresholds + T_int from random component only
alpha_frac  = sigma_random**2 / sigma_total**2    # random fraction of variance
T_int_eff   = alpha_frac * T_int_random           # effective T_int ≈ 3.9 s
T_req_stat  = (1.645**2) * 2 * (Iu_total**2) * T_int_eff / (0.10**2)
T_req_shed  = 5.0 * T_shed                        # 5 shedding cycles
T_req_wake_corrected = max(T_req_stat, T_req_shed)

print("\n" + "=" * 60)
print("THEORETICAL MINIMUM DWELL TIME")
print("=" * 60)
print(f"  Inlet  (95% CI, ε=5%)              : {T_req_inlet:.0f} s = {T_req_inlet/60:.1f} min")
print(f"  Wake naive (95% CI, ε=5%, T_int=T_shed): {T_req_wake_naive:.0f} s = {T_req_wake_naive/60:.0f} min  ← INFEASIBLE")
print(f"  Lag-1 bias on pure shedding        : {-dt/np.log(np.cos(2*np.pi*f_shed*dt)):.1f} s  (27× true T_shed/4)")
print(f"  T_int_eff (random component only)  : {T_int_eff:.2f} s")
print(f"  Wake corrected (90% CI, ε=10%, T_int_eff): {T_req_wake_corrected:.0f} s = {T_req_wake_corrected/60:.1f} min  ← FEASIBLE")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  ALGORITHM THRESHOLDS — WAKE MODE
# ─────────────────────────────────────────────────────────────────────────────
# Rationale for each relaxation is documented inline.

epsilon_ci       = 0.10   # Cond 1: 10% per-drone CI threshold (inlet: 5%)
                           # GPR smoothing recovers < 5% system accuracy
Z_score          = 1.645  # Cond 1: 90% confidence (inlet: 1.96 / 95%)

delta_stab       = 0.02   # Cond 2: 2% running-mean drift threshold (inlet: 1%)
                           # Wake mean can shift slightly across survey passes

stab_win_factor  = 8      # Cond 2: window = factor × T_int_est (inlet: 5)
                           # Longer window needed to capture slow vortex meander

N_eff_min        = 15     # Cond 3: minimum independent samples (inlet: 10)
                           # Need more samples to resolve high-Iu wake

N_shed_min       = 5      # Cond 4 (NEW): minimum complete shedding cycles
                           # Flushes periodic bias from running mean

# ── ACF estimator settings ────────────────────────────────────────────────────
ACF_BUFFER_SECS  = 4.0    # circular buffer holds this many T_shed worth of data
ACF_UPDATE_SECS  = 2.0    # re-estimate T_int every this many seconds
ACF_LAG_SECS     = 1.2    # max lag for ACF = this fraction × T_shed
                           # ensures we catch the first zero-crossing

acf_buffer_size  = int(ACF_BUFFER_SECS * T_shed * fs)   # ≈ 533 samples
acf_update_every = int(ACF_UPDATE_SECS * fs)             # 20 samples = 2 s
acf_lag_max      = int(ACF_LAG_SECS * T_shed * fs)      # ≈ 160 samples

# Burn-in: wait at least 3 shedding cycles before evaluating any condition
T_burnin = 3.0 * T_shed   # ≈ 40 s

# ─────────────────────────────────────────────────────────────────────────────
# 3.  SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

# ── Broadband AR(1) random turbulence ────────────────────────────────────────
alpha_ar  = np.exp(-dt / T_int_random)
noise_std = sigma_random * np.sqrt(1.0 - alpha_ar**2)
u_random  = np.zeros(N)
for i in range(1, N):
    u_random[i] = alpha_ar * u_random[i - 1] + noise_std * np.random.randn()

# ── Quasi-periodic vortex shedding ───────────────────────────────────────────
# Slow phase drift: shedding frequency is not perfectly constant in the
# atmospheric boundary layer. Modelled as a random walk in phase.
phase_noise = np.cumsum(np.random.randn(N) * 0.004 * 2.0 * np.pi * f_shed * dt)
u_shed = A_shed * np.sin(2.0 * np.pi * f_shed * time + phase_noise)

# ── Total wake velocity signal ────────────────────────────────────────────────
velocity = U_wake + u_random + u_shed

# ─────────────────────────────────────────────────────────────────────────────
# 4.  ONLINE ESTIMATORS
# ─────────────────────────────────────────────────────────────────────────────

def estimate_T_int_acf(buf_deque, dt, lag_max_samples):
    """
    Estimate the integral time scale T_int by integrating the sample
    autocorrelation function (ACF) from lag 0 up to its first zero-crossing.

    Why zero-crossing integration?
    ──────────────────────────────
    The lag-1 estimator assumes purely exponential ACF:
        ρ(τ) = exp(−τ/T_int)  →  T_int = −Δt / ln(ρ₁)
    In the wake, the ACF is a sum:
        ρ(τ) = α·exp(−τ/T_int_random) + (1−α)·cos(2πf_shed τ)
    The cosine term keeps ρ₁ ≈ 1, inflating T_int_lag1 to ~90 s (27×).
    Integrating only up to the first zero-crossing captures the exponential
    decay part and ignores the runaway periodic lobe, giving a faithful
    estimate of T_int_random (~3–5 s).

    Parameters
    ----------
    buf_deque   : deque containing recent velocity samples
    dt          : timestep [s]
    lag_max_samples : maximum lag to search (should be ≥ T_shed/4 · fs)

    Returns
    -------
    T_int_est : float [s], or None if buffer too small
    acf       : np.ndarray, normalised ACF for plotting
    """
    x = np.array(buf_deque)
    n = len(x)
    if n < 40:
        return None, None

    # Remove mean (necessary for unbiased covariance)
    x = x - np.mean(x)
    var = np.var(x)
    if var < 1e-12:
        return None, None

    # Sample ACF up to lag_max_samples
    lags = min(lag_max_samples, n // 3)
    acf = np.array([np.dot(x[:n - k], x[k:]) / ((n - k) * var)
                    for k in range(lags)])

    # Find first zero-crossing (linear interpolation for sub-sample accuracy)
    zero_lag = float(lags)    # default: no zero found within window
    for k in range(1, lags):
        if acf[k] <= 0.0:
            # Linear interpolation between k-1 and k
            zero_lag = (k - 1) + acf[k - 1] / (acf[k - 1] - acf[k])
            break

    # Integrate ACF from 0 to zero-crossing → T_int
    k_end = min(int(np.ceil(zero_lag)), lags - 1)
    # Include the partial sub-interval to the zero-crossing
    frac = zero_lag - int(zero_lag)
    partial = frac * acf[k_end] * dt if k_end < lags else 0.0
    T_int_est = float(np.trapezoid(acf[:k_end + 1], dx=dt)) + partial

    # Floor at 2 timesteps (numerical noise guard)
    T_int_est = max(T_int_est, 2.0 * dt)

    return T_int_est, acf


# ── Storage arrays for plotting ───────────────────────────────────────────────
running_mean     = np.zeros(N)
running_var      = np.zeros(N)
T_int_lag1_arr   = np.full(N, np.nan)   # biased lag-1 estimate (not used for stopping)
T_int_acf_arr    = np.full(N, np.nan)   # corrected ACF zero-crossing estimate
ci_rel_arr       = np.full(N, np.nan)
stability_arr    = np.full(N, np.nan)
n_cycles_arr     = np.full(N, np.nan)

cond1_arr        = np.zeros(N, dtype=bool)
cond2_arr        = np.zeros(N, dtype=bool)
cond3_arr        = np.zeros(N, dtype=bool)
cond4_arr        = np.zeros(N, dtype=bool)

stop_idx  = None
hover_time = None

# ── State variables ───────────────────────────────────────────────────────────
M2            = 0.0            # Welford M2 accumulator for variance
T_int_used    = T_int_random   # initial T_int guess — updated online
last_acf_step = 0

# Running lag-1 covariance (online, O(1) per step, for comparison plot)
lag1_sum = 0.0                 # Σ u'_t · u'_{t-1}
lag1_n   = 0
v_prev_demean = 0.0

signal_buf = deque(maxlen=acf_buffer_size)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN ONLINE SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
for i in range(1, N):
    v             = velocity[i]
    current_time  = time[i]
    n_samples     = i + 1

    # ── Welford running mean and variance ─────────────────────────────────────
    old_mean      = running_mean[i - 1]
    delta         = v - old_mean
    running_mean[i] = old_mean + delta / n_samples
    M2           += delta * (v - running_mean[i])
    running_var[i]  = M2 / n_samples    # population variance

    # ── Running lag-1 covariance (online) ────────────────────────────────────
    v_demean = v - running_mean[i]
    if i >= 2:
        lag1_sum += v_demean * v_prev_demean
        lag1_n   += 1
    v_prev_demean = v_demean

    # ── ACF-based T_int update ────────────────────────────────────────────────
    signal_buf.append(v)
    if (i - last_acf_step >= acf_update_every) and (len(signal_buf) > 40):
        T_new, _ = estimate_T_int_acf(signal_buf, dt, acf_lag_max)
        if T_new is not None:
            # Exponential moving average to smooth the estimate
            T_int_used = 0.7 * T_int_used + 0.3 * T_new
        last_acf_step = i

    # ── Lag-1 T_int (for plot only — biased in wake) ──────────────────────────
    if lag1_n > 10 and running_var[i] > 1e-12:
        rho1_hat = (lag1_sum / lag1_n) / running_var[i]
        rho1_hat = np.clip(rho1_hat, 0.001, 0.9999)
        T_int_lag1_arr[i] = -dt / np.log(rho1_hat)
    elif i > 0:
        T_int_lag1_arr[i] = T_int_lag1_arr[i - 1]

    T_int_acf_arr[i] = T_int_used

    # ── Burn-in guard ─────────────────────────────────────────────────────────
    if current_time <= T_burnin:
        continue

    # ── Adaptive stability window ─────────────────────────────────────────────
    stab_window_samples = max(int(stab_win_factor * T_int_used * fs), 40)

    # ── Condition 3: Minimum independent samples ──────────────────────────────
    N_eff  = current_time / (2.0 * T_int_used)
    cond3  = N_eff >= N_eff_min
    cond3_arr[i] = cond3

    # ── Condition 4: Minimum shedding cycles flushed ──────────────────────────
    n_cycles = current_time / T_shed
    n_cycles_arr[i] = n_cycles
    cond4  = n_cycles >= N_shed_min
    cond4_arr[i] = cond4

    # ── Condition 1: Statistical confidence interval ───────────────────────────
    sigma_mean = np.sqrt(2.0 * running_var[i] * T_int_used / current_time)
    ci_abs     = Z_score * sigma_mean
    ci_rel_now = ci_abs / max(abs(running_mean[i]), 1e-6)
    ci_rel_arr[i] = ci_rel_now
    cond1  = ci_rel_now < epsilon_ci
    cond1_arr[i] = cond1

    # ── Condition 2: Running mean stability ───────────────────────────────────
    if i >= stab_window_samples:
        drift = abs(running_mean[i] - running_mean[i - stab_window_samples])
        stab  = drift / max(abs(running_mean[i]), 1e-6)
    else:
        stab = np.inf
    stability_arr[i] = stab
    cond2  = stab < delta_stab
    cond2_arr[i] = cond2

    # ── All four conditions must be simultaneously true ────────────────────────
    if cond1 and cond2 and cond3 and cond4 and stop_idx is None:
        stop_idx   = i
        hover_time = current_time

# ─────────────────────────────────────────────────────────────────────────────
# 6.  RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALGORITHM RESULTS")
print("=" * 60)
if hover_time is not None:
    est  = running_mean[stop_idx]
    true = U_wake
    err  = abs(est - true) / true
    print(f"  Stopped at:          {hover_time:.1f} s  = {hover_time/60:.2f} min")
    print(f"  7-min budget used:   {hover_time/420*100:.0f}%")
    print(f"  Estimated mean:      {est:.3f} m/s")
    print(f"  True wake mean:      {true:.3f} m/s  (unknown to algorithm)")
    print(f"  Absolute error:      {abs(est-true):.3f} m/s")
    print(f"  Relative error:      {err*100:.2f}%  (within {epsilon_ci*100:.0f}% target)")
    print(f"  T_int (ACF) at stop: {T_int_acf_arr[stop_idx]:.2f} s")
    print(f"  N_eff at stop:       {hover_time/(2*T_int_acf_arr[stop_idx]):.1f}")
    print(f"  Shedding cycles:     {hover_time/T_shed:.1f}")
else:
    print("  WARNING: Did not converge within simulation window.")
    print("  Increase T_total or further relax thresholds.")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
BUDGET_TIME = 7.0 * 60.0    # 7-min hard budget [s]
STOP_COLOR  = '#2ca02c'      # green
BUDGET_COLOR = '#ff7f0e'     # orange

fig = plt.figure(figsize=(14, 13))
fig.suptitle(
    f'LAWSS Wake Sampling Algorithm  |  DCMA Mode\n'
    f'$D$ = {D_char:.0f} m,  $U_{{inlet}}$ = {U_inlet:.0f} m/s,  '
    f'$f_{{shed}}$ = {f_shed:.3f} Hz,  $T_{{shed}}$ = {T_shed:.1f} s,  '
    f'$I_u$ = {Iu_total*100:.0f}%',
    fontsize=12, fontweight='bold', y=0.99
)
gs = gridspec.GridSpec(4, 1, hspace=0.40, top=0.93, bottom=0.06)

def vline(ax, x, color, style='--', lw=1.5, zorder=5):
    ax.axvline(x, color=color, ls=style, lw=lw, zorder=zorder)

# ── Panel 1: Velocity signal ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot(time, velocity, color='steelblue', lw=0.5, alpha=0.45,
         label='Instantaneous velocity $u(t)$')
ax1.plot(time, running_mean, color='crimson', lw=1.8,
         label=r'Running mean $\hat{U}_T$')
ax1.axhline(U_wake, color='black', ls='--', lw=1.5,
            label=f'True wake mean $U_{{wake}}$ = {U_wake:.2f} m/s (unknown)')
if hover_time:
    vline(ax1, hover_time, STOP_COLOR)
    ax1.annotate(f'STOP\n{hover_time:.0f}s\n({hover_time/60:.1f} min)',
                 xy=(hover_time, np.nanmax(velocity)),
                 xytext=(hover_time + 15, np.nanmax(velocity)),
                 fontsize=8, color=STOP_COLOR, va='top')
vline(ax1, BUDGET_TIME, BUDGET_COLOR, style=':', lw=2.0)
ax1.axvspan(0, T_burnin, alpha=0.07, color='gray', label=f'Burn-in ({T_burnin:.0f}s)')
ax1.set_ylabel('Velocity [m/s]', fontsize=9)
ax1.set_title(
    f'Wake velocity  |  $I_u^{{random}}$ = {Iu_random*100:.0f}%,  '
    f'Shedding amplitude = {A_shed:.1f} m/s  →  $I_u^{{total}}$ ≈ {Iu_total*100:.0f}%',
    fontsize=9)
ax1.legend(fontsize=7.5, ncol=2, loc='upper right')
ax1.grid(alpha=0.25)

# ── Panel 2: T_int estimates comparison ───────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
valid_lag1 = ~np.isnan(T_int_lag1_arr)
valid_acf  = ~np.isnan(T_int_acf_arr)
ax2.semilogy(time[valid_lag1], T_int_lag1_arr[valid_lag1],
             color='tomato', lw=1.2, alpha=0.8,
             label=r'$\hat{\mathcal{T}}_{lag1}$  — standard lag-1 estimator (BIASED in wake)')
ax2.semilogy(time[valid_acf], T_int_acf_arr[valid_acf],
             color='purple', lw=2.0,
             label=r'$\hat{\mathcal{T}}_{ACF}$  — zero-crossing integral (WAKE-CORRECT)')
ax2.axhline(T_int_random, color='navy', ls=':', lw=1.5,
            label=f'True $\\mathcal{{T}}_{{random}}$ = {T_int_random:.0f} s')
ax2.axhline(T_shed, color='red', ls=':', lw=1.2,
            label=f'$T_{{shed}}$ = {T_shed:.1f} s  (lag-1 converges toward this, not T_int!)')
ax2.axhline(T_shed / 4.0, color='green', ls='-.', lw=1.2,
            label=f'$T_{{shed}}/4$ = {T_shed/4:.1f} s  (ACF first zero-crossing expected location)')
if hover_time:
    vline(ax2, hover_time, STOP_COLOR)
vline(ax2, BUDGET_TIME, BUDGET_COLOR, style=':', lw=2.0)
ax2.axvspan(0, T_burnin, alpha=0.07, color='gray')
ax2.set_ylabel(r'$\hat{\mathcal{T}}$ [s]', fontsize=9)
ax2.set_title(
    'Integral time scale estimation: lag-1 (biased) vs. ACF zero-crossing (corrected)',
    fontsize=9)
ax2.legend(fontsize=7, ncol=1, loc='upper right')
ax2.set_ylim([0.5, 200])
ax2.grid(alpha=0.25, which='both')

# ── Panel 3: Conditions 1, 3, 4 ───────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
valid_ci = ~np.isnan(ci_rel_arr)
ax3.semilogy(time[valid_ci], ci_rel_arr[valid_ci],
             color='darkorange', lw=1.4,
             label=f'$Z_{{90\\%}} \\cdot \\sigma_{{\\bar{{U}}}} / \\hat{{U}}_T$  (Cond. 1: must be < {epsilon_ci*100:.0f}%)')
ax3.axhline(epsilon_ci, color='darkorange', ls='--', lw=1.5,
            label=f'Cond. 1 threshold  ε = {epsilon_ci*100:.0f}%  (90% CI, relaxed from 5%)')

# Shade regions where conditions 3 & 4 are met
t_cond3_met = next((time[i] for i in range(N) if cond3_arr[i]), None)
t_cond4_met = N_shed_min * T_shed
if t_cond3_met:
    vline(ax3, t_cond3_met, 'teal', style='-.', lw=1.2)
    ax3.annotate(f'Cond.3 met\n$N_{{eff}}≥{N_eff_min}$\n({t_cond3_met:.0f}s)',
                 xy=(t_cond3_met, epsilon_ci * 0.7), fontsize=7, color='teal', ha='right')
vline(ax3, t_cond4_met, 'magenta', style='-.', lw=1.2)
ax3.annotate(f'Cond.4 met\n{N_shed_min} shed. cycles\n({t_cond4_met:.0f}s)',
             xy=(t_cond4_met, epsilon_ci * 1.5), fontsize=7, color='magenta', ha='left')
if hover_time:
    vline(ax3, hover_time, STOP_COLOR)
vline(ax3, BUDGET_TIME, BUDGET_COLOR, style=':', lw=2.0)
ax3.axvspan(0, T_burnin, alpha=0.07, color='gray')
ax3.set_ylabel('Relative CI  [—]', fontsize=9)
ax3.set_title(
    f'Conditions 1, 3, 4 — CI convergence and minimum dwell gates',
    fontsize=9)
ax3.legend(fontsize=7.5, loc='upper right')
ax3.grid(alpha=0.25, which='both')

# ── Panel 4: Condition 2 — Stability ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[3])
valid_stab = ~np.isnan(stability_arr) & (stability_arr > 0)
ax4.semilogy(time[valid_stab], stability_arr[valid_stab],
             color='teal', lw=1.2, alpha=0.9,
             label=f'Mean drift over $8 \\times \\hat{{\\mathcal{{T}}}}_{{ACF}}$ window  (Cond. 2)')
ax4.axhline(delta_stab, color='teal', ls='--', lw=1.5,
            label=f'Cond. 2 threshold  δ = {delta_stab*100:.0f}%  (relaxed from 1%)')
if hover_time:
    vline(ax4, hover_time, STOP_COLOR)
    ax4.annotate(
        f'STOP  t = {hover_time:.0f}s\n'
        f'Error = {abs(running_mean[stop_idx]-U_wake)/U_wake*100:.1f}%\n'
        f'Budget used = {hover_time/420*100:.0f}%',
        xy=(hover_time, delta_stab),
        xytext=(hover_time + 10, delta_stab * 3),
        fontsize=8.5, color=STOP_COLOR,
        arrowprops=dict(arrowstyle='->', color=STOP_COLOR, lw=1.2)
    )
vline(ax4, BUDGET_TIME, BUDGET_COLOR, style=':', lw=2.0)
ax4.axvspan(0, T_burnin, alpha=0.07, color='gray',
            label=f'Burn-in: {T_burnin:.0f}s (3 shedding cycles)')
ax4.set_ylabel('Relative drift  [—]', fontsize=9)
ax4.set_xlabel('Sampling time  [s]', fontsize=10)
ax4.set_title(
    'Condition 2 — Running mean stability (extended window for wake meander)',
    fontsize=9)
ax4.legend(fontsize=7.5, loc='upper right')
ax4.grid(alpha=0.25, which='both')

# Shared vertical line labels
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlim([0, T_total])

# Legend for shared vertical lines
from matplotlib.lines import Line2D
shared_handles = [
    Line2D([0], [0], color=STOP_COLOR, ls='--', lw=1.5,
           label=f'Algorithm stop  t = {hover_time:.0f}s' if hover_time else 'No stop'),
    Line2D([0], [0], color=BUDGET_COLOR, ls=':', lw=2.0,
           label='7-min battery budget  t = 420s'),
]
fig.legend(handles=shared_handles, loc='lower center', ncol=2, fontsize=9,
           frameon=True, bbox_to_anchor=(0.5, 0.01))

plt.savefig('wake_sampling_algorithm.png',
            dpi=600, bbox_inches='tight')
plt.close()
print("\nPlot saved: wake_sampling_algorithm.png")
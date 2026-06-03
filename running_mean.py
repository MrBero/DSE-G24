import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # change to 'Qt5Agg' or 'Agg' if needed
import matplotlib.pyplot as plt
plt.style.use('dark_background')

U_mean  = 5.0       # mean inlet wind speed  [m/s]
I_u     = 0.15      # turbulence intensity  [-]   typical ABL: 0.10–0.25
T_int   = 10.0      # integral time scale  [s]    typical ABL: 5–30 s
dt      = 0.1       # sampling interval  [s]      (= 1 / sample_rate)
epsilon = 0.02      # convergence threshold  [-]  (2 % of mean)
T_max   = 300.0     # max hover time safety cutoff  [s]
seed    = 42        # for reproducibility; set None for random each run
n_trials = 5        # overlay this many independent realisations

sigma_u   = I_u * U_mean                        # turbulence std dev  [m/s]
phi       = np.exp(-dt / T_int)                  # AR(1) coefficient
sigma_eps = sigma_u * np.sqrt(1 - phi**2)        # AR(1) innovation std dev
N_max     = int(T_max / dt)

T_min_theory = 2.0 * I_u**2 * T_int / epsilon**2   # theoretical minimum [s]

print("=" * 60)
print("  RUNNING AVERAGE CONVERGENCE — CONFIGURATION SUMMARY")
print("=" * 60)
print(f"  U_mean     = {U_mean:.2f} m/s")
print(f"  I_u        = {I_u:.3f}   (sigma_u = {sigma_u:.3f} m/s)")
print(f"  T_int      = {T_int:.1f} s")
print(f"  dt         = {dt:.3f} s   (sample rate = {1/dt:.1f} Hz)")
print(f"  epsilon    = {epsilon*100:.1f} %")
print(f"  T_max      = {T_max:.0f} s")
print(f"  Theoretical T_min = {T_min_theory:.1f} s")
print("=" * 60)

def generate_turbulence(N, rng):
    u_prime = np.zeros(N)
    eps = rng.normal(0.0, sigma_eps, N)
    for n in range(1, N):
        u_prime[n] = phi * u_prime[n - 1] + eps[n]
    return U_mean + u_prime          # total signal

def run_trial(u_signal):
    N = len(u_signal)
    times   = np.arange(1, N + 1) * dt
    means   = np.zeros(N)
    metrics = np.full(N, np.nan)

    running_sum    = 0.0
    running_sum_sq = 0.0
    # For lag-1 autocorrelation (Welford-style)
    prev           = u_signal[0]
    cross_sum      = 0.0      # sum of u[k]*u[k-1]
    sq_sum         = 0.0      # sum of u[k]^2

    t_conv = None

    for n in range(N):
        u = u_signal[n]
        running_sum    += u
        running_sum_sq += u * u
        if n > 0:
            cross_sum += u * prev
            sq_sum    += u * u
        prev = u

        mean_n = running_sum / (n + 1)
        means[n] = mean_n

        # Need at least a small window before estimating dispersion
        if n < 10:
            continue

        # Unbiased variance estimate of the raw signal
        var_u = (running_sum_sq / (n + 1)) - mean_n**2
        var_u = max(var_u, 1e-12)

        # Lag-1 autocorrelation estimate -> integral time scale
        # rho_1 = cov(u[n], u[n-1]) / var(u)
        # For AR(1):  T_int_est = -dt / ln(rho_1)
        if n >= 2:
            mean_cross = cross_sum / n          # E[u[k]*u[k-1]]
            rho1 = (mean_cross - mean_n**2) / var_u
            rho1 = np.clip(rho1, 1e-6, 1.0 - 1e-6)
            T_int_est = -dt / np.log(rho1)
        else:
            T_int_est = T_int   # fall back to prior before we have data

        # Variance of the running mean (from the derivation)
        T_elapsed = (n + 1) * dt
        var_mean  = (2.0 * var_u * T_int_est) / T_elapsed
        sigma_mean = np.sqrt(max(var_mean, 0.0))

        metric = sigma_mean / abs(mean_n) if abs(mean_n) > 1e-9 else np.inf
        metrics[n] = metric

        # Convergence check
        if t_conv is None and metric < epsilon:
            t_conv = T_elapsed

    return t_conv, times, means, metrics

rng = np.random.default_rng(seed)
results = []
conv_times = []

for trial in range(n_trials):
    u = generate_turbulence(N_max, rng)
    t_conv, times, means, metrics = run_trial(u)
    results.append((times, means, metrics, u))
    conv_times.append(t_conv)
    status = f"{t_conv:.1f} s" if t_conv is not None else "NOT CONVERGED"
    print(f"  Trial {trial+1:2d}: converged at {status}")

print("=" * 60)
conv_valid = [t for t in conv_times if t is not None]
if conv_valid:
    print(f"  Mean convergence time : {np.mean(conv_valid):.1f} s")
    print(f"  Std  convergence time : {np.std(conv_valid):.1f} s")
    print(f"  Theoretical T_min     : {T_min_theory:.1f} s")
print("=" * 60)

def autocorr(x, max_lag):
    x = x - x.mean()
    var = np.var(x)
    lags = np.arange(max_lag)
    ac = np.array([np.mean(x[:len(x)-k] * x[k:]) for k in lags]) / var
    return lags * dt, ac

PALETTE = ['#00d4ff', '#ff6b6b', '#a8ff78', '#f7971e', '#d4fc79',
           '#f953c6', '#b91d73', '#4facfe', '#43e97b']

fig = plt.figure(figsize=(16, 9), constrained_layout=True)
fig.suptitle('Running Average Convergence — Inlet Velocity Measurement',
             fontsize=14, fontweight='bold', color='white')

gs = fig.add_gridspec(3, 2)
ax_sig  = fig.add_subplot(gs[0, :])    # full-width: raw signal(s)
ax_mean = fig.add_subplot(gs[1, :])    # full-width: running mean
ax_met  = fig.add_subplot(gs[2, 0])   # convergence metric
ax_acf  = fig.add_subplot(gs[2, 1])   # autocorrelation function

# --- Raw signal (first trial only for clarity, others faint) ---
for i, (times, means, metrics, u) in enumerate(results):
    alpha = 0.8 if i == 0 else 0.18
    lw    = 0.7 if i == 0 else 0.4
    ax_sig.plot(times, u, color=PALETTE[i % len(PALETTE)], lw=lw, alpha=alpha)

ax_sig.axhline(U_mean, color='white', lw=1.2, ls='--', label=f'True mean = {U_mean} m/s')
ax_sig.set_ylabel('u(t)  [m/s]')
ax_sig.set_title('Raw velocity signal (AR(1) turbulence synthesis)', fontsize=10)
ax_sig.legend(fontsize=8)
ax_sig.set_xlim(0, times[-1])

# --- Running mean ---
for i, (times, means, metrics, u) in enumerate(results):
    ax_mean.plot(times, means, color=PALETTE[i % len(PALETTE)], lw=1.0, alpha=0.9,
                 label=f'Trial {i+1}' if i < 3 else '_')
    if conv_times[i] is not None:
        ax_mean.axvline(conv_times[i], color=PALETTE[i % len(PALETTE)],
                        ls=':', lw=1.2, alpha=0.7)

ax_mean.axhline(U_mean, color='white', lw=1.2, ls='--')
ax_mean.axvline(T_min_theory, color='yellow', lw=1.5, ls='-.',
                label=f'Theoretical T_min = {T_min_theory:.1f} s')

# Theoretical envelope: mean ± epsilon * U_mean
T_arr  = np.linspace(dt * 10, T_max, 400)
envelope = epsilon * U_mean * np.sqrt(T_arr / (2.0 * I_u**2 * T_int)) / np.sqrt(T_arr / (2.0 * I_u**2 * T_int))
# Correctly: sigma_Uhat = sqrt(2*sigma_u^2*T_int/T)
env_sigma = np.sqrt(2.0 * sigma_u**2 * T_int / T_arr)
ax_mean.fill_between(T_arr, U_mean - env_sigma, U_mean + env_sigma,
                     color='cyan', alpha=0.12, label='±σ_mean envelope (theory)')
ax_mean.fill_between(T_arr, U_mean - epsilon*U_mean, U_mean + epsilon*U_mean,
                     color='yellow', alpha=0.08, label=f'±{epsilon*100:.0f}% band')

ax_mean.set_ylabel('Running mean  [m/s]')
ax_mean.set_title('Running mean convergence', fontsize=10)
ax_mean.legend(fontsize=7, ncol=3)
ax_mean.set_xlim(0, times[-1])

# --- Convergence metric ---
for i, (times, means, metrics, u) in enumerate(results):
    ax_met.plot(times, metrics * 100, color=PALETTE[i % len(PALETTE)],
                lw=0.9, alpha=0.85)
    if conv_times[i] is not None:
        ax_met.axvline(conv_times[i], color=PALETTE[i % len(PALETTE)],
                       ls=':', lw=1.2, alpha=0.7)

# Theoretical decay curve
met_theory = np.sqrt(2.0 * sigma_u**2 * T_int / T_arr) / U_mean * 100
ax_met.plot(T_arr, met_theory, 'w--', lw=1.3, label='Theory: σ_mean/U (%)' )
ax_met.axhline(epsilon * 100, color='yellow', lw=1.5, ls='-.', label=f'Threshold {epsilon*100:.0f}%')
ax_met.axvline(T_min_theory, color='yellow', lw=1.2, ls='-.', alpha=0.6)
ax_met.set_xlabel('Time  [s]')
ax_met.set_ylabel('σ_mean / U_mean  [%]')
ax_met.set_title('Convergence metric', fontsize=10)
ax_met.legend(fontsize=7)
ax_met.set_xlim(0, times[-1])
ax_met.set_ylim(0, epsilon * 100 * 8)

# --- Autocorrelation function (first trial) ---
max_lag_s = min(5 * T_int, T_max / 4)
max_lag_n = int(max_lag_s / dt)
lag_times, ac = autocorr(results[0][3], max_lag_n)   # u signal from trial 0

tau_theory = np.linspace(0, max_lag_s, 200)
ac_theory  = np.exp(-tau_theory / T_int)

ax_acf.plot(lag_times, ac, color=PALETTE[0], lw=1.0, label='Estimated ACF (trial 1)')
ax_acf.plot(tau_theory, ac_theory, 'w--', lw=1.3, label=f'Theory: exp(-τ/T_int)')
ax_acf.axvline(T_int, color='yellow', lw=1.2, ls='-.', label=f'T_int = {T_int:.1f} s')
ax_acf.axhline(0, color='white', lw=0.5, ls=':')
ax_acf.fill_between(lag_times, ac, 0, where=(ac > 0),
                    color=PALETTE[0], alpha=0.15, label='Area = T_int (approx)')
ax_acf.set_xlabel('Lag τ  [s]')
ax_acf.set_ylabel('ρ(τ)')
ax_acf.set_title('Autocorrelation function', fontsize=10)
ax_acf.legend(fontsize=7)
ax_acf.set_xlim(0, max_lag_s)
ax_acf.set_ylim(-0.3, 1.05)

plt.savefig('running_average_convergence.png', dpi=150, bbox_inches='tight')
print("\nFigure saved to running_average_convergence.png")
plt.show()
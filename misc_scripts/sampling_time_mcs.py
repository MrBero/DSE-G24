import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================================== #
#  CORE SIMULATION ENGINES                                                   #
# =========================================================================== #

def acf_zero_crossing_T_int(buf, dt, max_lag):
    n = len(buf)
    if n < max_lag + 2: return np.nan
    x = buf - buf.mean()
    var = np.var(x)
    if var < 1e-12: return np.nan
    
    lags = np.arange(max_lag + 1)
    acf = np.array([np.mean(x[:n-k] * x[k:]) for k in lags]) / var
    
    sign_changes = np.where(np.diff(np.sign(acf)))[0]
    if len(sign_changes) == 0: return np.nan
    
    k0 = sign_changes[0]
    frac = acf[k0] / (acf[k0] - acf[k0 + 1])
    
    tau_grid = lags[:k0 + 1] * dt
    T_est = float(np.trapezoid(acf[:k0 + 1], tau_grid))
    return max(T_est, dt)

def sim_inlet(seed, I_u, epsilon_ci, U_mean=12.0, T_int=5.0, fs=10.0, T_total=600):
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    N = int(T_total * fs)
    sigma_u = I_u * U_mean
    phi = np.exp(-dt / T_int)
    sigma_eps = sigma_u * np.sqrt(1 - phi**2)
    
    u_prime = np.zeros(N)
    eps = rng.normal(0.0, sigma_eps, N)
    for n in range(1, N):
        u_prime[n] = phi * u_prime[n-1] + eps[n]
    velocity = U_mean + u_prime
    
    wf_n = 0; wf_mean = 0.0; wf_M2 = 0.0
    L_n = 0; L_mux = 0.0; L_muy = 0.0; L_mxy = 0.0; L_mx2 = 0.0
    T_int_est = T_int
    ema_alpha = 0.05
    
    mean_history = np.full(N, np.nan)
    STAB_WIN = int(5.0 * T_int * fs)
    
    for n in range(N):
        u = velocity[n]
        prev = velocity[n-1] if n > 0 else u
        
        wf_n += 1
        d = u - wf_mean
        wf_mean += d / wf_n
        wf_M2 += d * (u - wf_mean)
        
        if n > 0:
            L_n += 1
            L_mux += (u - L_mux) / L_n
            L_muy += (prev - L_muy) / L_n
            L_mxy += (u*prev - L_mxy) / L_n
            L_mx2 += (u*u - L_mx2) / L_n
            
            if L_n > 20:
                cov_lag = L_mxy - L_mux * L_muy
                var_lag = max(L_mx2 - L_mux**2, 1e-12)
                rho1 = np.clip(cov_lag / var_lag, 1e-6, 1 - 1e-6)
                T_new = -dt / np.log(rho1)
                T_int_est = ema_alpha * T_new + (1 - ema_alpha) * T_int_est
                
        mean_history[n] = wf_mean
        
        if n < int(5.0 * T_int * fs): continue
        
        current_T = (n + 1) * dt
        var_u = wf_M2 / max(wf_n - 1, 1)
        sigma_Ubar = np.sqrt(max(2.0 * var_u * T_int_est / current_T, 0.0))
        
        N_eff = current_T / (2.0 * T_int_est)
        ci_rel = 1.96 * sigma_Ubar / max(abs(wf_mean), 1e-9)
        
        oldest = mean_history[n - STAB_WIN] if n >= STAB_WIN else np.nan
        stab = abs(wf_mean - oldest) / max(abs(wf_mean), 1e-9) if np.isfinite(oldest) else np.inf
        
        if (ci_rel < epsilon_ci) and (stab < 0.01) and (N_eff >= 10):
            return current_T
            
    return T_total

def sim_wake(seed, I_u_rand, epsilon_ci, U_inf=14.5, U_wake=9.0, I_u_shed=0.15, T_int_rand=5.0, St=0.10, D=20.0, fs=10.0, T_total=420):
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    N = int(T_total * fs)
    
    sigma_rand = I_u_rand * U_wake
    sigma_shed = I_u_shed * U_wake
    f_shed = St * U_inf / D
    T_shed = 1.0 / f_shed
    
    phi_rand = np.exp(-dt / T_int_rand)
    sigma_eps = sigma_rand * np.sqrt(1 - phi_rand**2)
    
    u_rand = np.zeros(N)
    eps = rng.normal(0.0, sigma_eps, N)
    for n in range(1, N):
        u_rand[n] = phi_rand * u_rand[n-1] + eps[n]
        
    time_arr = np.arange(N) * dt
    u_shed = sigma_rand * np.sqrt(2) * np.sin(2 * np.pi * f_shed * time_arr + rng.uniform(0, 2 * np.pi))
    velocity = U_wake + u_rand + u_shed
    
    wf_n = 0; wf_mean = 0.0; wf_M2 = 0.0
    acf_buf_len = max(int(4 * T_shed * fs), 400)
    acf_max_lag = max(int(1.5 * T_shed * fs), 150)
    acf_buffer = np.zeros(acf_buf_len)
    
    alpha_frac = sigma_rand**2 / (sigma_rand**2 + sigma_shed**2)
    T_int_eff_cur = alpha_frac * T_int_rand
    
    mean_history = np.full(N, np.nan)
    
    for n in range(N):
        u = velocity[n]
        wf_n += 1
        d = u - wf_mean
        wf_mean += d / wf_n
        wf_M2 += d * (u - wf_mean)
        
        mean_history[n] = wf_mean
        
        acf_buffer[n % acf_buf_len] = u
        if n > 0 and n % 50 == 0 and n >= acf_buf_len:
            start = n % acf_buf_len
            ordered = np.concatenate([acf_buffer[start:], acf_buffer[:start]])
            T_new = acf_zero_crossing_T_int(ordered, dt, acf_max_lag)
            if np.isfinite(T_new) and T_new > 0:
                T_int_eff_cur = 0.3 * T_new + 0.7 * T_int_eff_cur
                
        current_T = (n + 1) * dt
        if current_T < max(5 * T_int_eff_cur, 5 * T_shed): continue
        
        var_u = wf_M2 / max(wf_n - 1, 1)
        sigma_Ubar = np.sqrt(max(2 * var_u * T_int_eff_cur / current_T, 0.0))
        
        N_eff = current_T / (2 * T_int_eff_cur)
        ci_rel = 1.645 * sigma_Ubar / abs(wf_mean) if abs(wf_mean) > 1e-9 else np.inf
        
        oldest = mean_history[n - int(8 * T_int_eff_cur * fs)] if n >= int(8 * T_int_eff_cur * fs) else np.nan
        stab = abs(wf_mean - oldest) / abs(wf_mean) if np.isfinite(oldest) else np.inf
        
        if (ci_rel < epsilon_ci) and (stab < 0.02) and (N_eff >= 15) and (current_T / T_shed >= 5):
            return current_T
            
    return T_total 

# =========================================================================== #
#  SWEEP EXECUTION AND PLOTTING                                              #
# =========================================================================== #
if __name__ == "__main__":
    n_seeds = 40  # Balanced for speed and statistical significance
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ----------------------------------------------------------------------- #
    # 1. INLET SAMPLING PARAMETER SWEEP
    # ----------------------------------------------------------------------- #
    print("Executing Inlet Plane Parameter Sweep...")
    I_u_range = np.linspace(0.03, 0.18, 8)
    epsilon_inlet_modes = [0.03, 0.05, 0.08]
    colors_inlet = ['#ff4757', '#2e92ff', '#2ed573']
    
    for eps, color in zip(epsilon_inlet_modes, colors_inlet):
        mean_times = []
        p95_times = []
        for I_u in I_u_range:
            times = [sim_inlet(seed, I_u, eps) for seed in range(n_seeds)]
            mean_times.append(np.mean(times))
            p95_times.append(np.percentile(times, 95))
            
        ax1.plot(I_u_range * 100, mean_times, color=color, linestyle='-', linewidth=2, 
                 label=f'Mean (ε = {eps*100:.0f}%)')
        ax1.plot(I_u_range * 100, p95_times, color=color, linestyle='--', linewidth=1.5, 
                 label=f'95th Pct (ε = {eps*100:.0f}%)')
        
    ax1.set_title('Inlet Plane Convergence Trends', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Turbulence Intensity ($I_u$) [%]', fontsize=11)
    ax1.set_ylabel('Required Sampling Time [s]', fontsize=11)
    ax1.set_ylim(0, 450)
    ax1.legend(fontsize=9, loc='upper left', ncol=1)
    
    # ----------------------------------------------------------------------- #
    # 2. WAKE SAMPLING PARAMETER SWEEP
    # ----------------------------------------------------------------------- #
    print("Executing Wake Plane Parameter Sweep...")
    I_u_wake_range = np.linspace(0.12, 0.38, 8)
    epsilon_wake_modes = [0.05, 0.10, 0.15]
    colors_wake = ['#ffa502', '#9b5de5', '#00bbf9']
    
    for eps, color in zip(epsilon_wake_modes, colors_wake):
        mean_times = []
        p95_times = []
        for I_u_rand in I_u_wake_range:
            times = [sim_wake(seed, I_u_rand, eps) for seed in range(n_seeds)]
            mean_times.append(np.mean(times))
            p95_times.append(np.percentile(times, 95))
            
        ax2.plot(I_u_wake_range * 100, mean_times, color=color, linestyle='-', linewidth=2, 
                 label=f'Mean (ε = {eps*100:.0f}%)')
        ax2.plot(I_u_wake_range * 100, p95_times, color=color, linestyle='--', linewidth=1.5, 
                 label=f'95th Pct (ε = {eps*100:.0f}%)')
        
    ax2.axhline(420, color='red', linestyle=':', linewidth=1.5, label='7-Min Drone Battery Cap')
    ax2.set_title('Wake Plane Convergence Trends (DCMA Mode)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Random Turbulence Intensity ($I_{u,rand}$) [%]', fontsize=11)
    ax2.set_ylabel('Required Sampling Time [s]', fontsize=11)
    ax2.set_ylim(0, 450)
    ax2.legend(fontsize=9, loc='upper left', ncol=1)
    
    plt.tight_layout()
    plt.show()
"""
Improved PINN for steady 2D incompressible flow around a cylinder.

Inverse problem / data assimilation:
    Given ~150 sparse (u, v) sensor measurements + Navier-Stokes,
    reconstruct the full velocity *and* pressure field.

Key improvements over the baseline:
  1. Stream-function formulation: net outputs (psi, p), velocities are
     u =  d psi / d y,  v = -d psi / d x. Continuity is then EXACT.
  2. KMeans-stratified sensor placement (vs. random sampling, which
     leaves big gaps at N=150).
  3. Inputs normalized to [-1, 1]; outputs in physical units (avoids
     contaminating the autograd graph).
  4. 6 x 50 tanh MLP, Xavier init.
  5. Explicit no-slip BC on cylinder surface.
  6. Physics-loss weight ramped 0.01 -> 1.0 across the first 60% of
     Adam (cures the PINN cold-start pathology).
  7. Adam (5000) -> L-BFGS (max 2000), with FIXED collocation points
     during L-BFGS so the line search behaves.
  8. Pure CPU (no CUDA), threading set explicitly.

Dependencies (CPU-only, all pip-installable):
    torch  pandas  numpy  matplotlib  scikit-learn  scipy  openpyxl
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

# ============================================================
# 0. Config & reproducibility (CPU only)
# ============================================================
SEED            = 42
N_SENSORS       = 150       # sparse "drone" measurements
N_BC            = 200       # cylinder-surface points (no-slip)
N_COLLOC_ADAM   = 2000      # resampled per Adam epoch
N_COLLOC_LBFGS  = 4000      # FIXED for L-BFGS line search
ADAM_EPOCHS     = 5000
LBFGS_MAX_ITER  = 2000
LAM_BC          = 1.0
NU              = 0.01      # kinematic viscosity -- replace w/ your CFD value
LR              = 1e-3
HIDDEN          = (50, 50, 50, 50, 50, 50)   # 6 hidden layers, 50 neurons
ACTIVATION      = nn.Tanh

torch.manual_seed(SEED); np.random.seed(SEED)
torch.set_default_dtype(torch.float32)
torch.set_num_threads(max(1, torch.get_num_threads()))
DEVICE = torch.device("cpu")

# ============================================================
# 1. Load CFD data (preserves the original cylinder/domain logic)
# ============================================================
print("Loading data...")
df_field = pd.read_excel("field.xlsx")
df_cyl   = pd.read_excel("cylinder (1).xlsx")
df_field.columns = df_field.columns.str.strip()
df_cyl.columns   = df_cyl.columns.str.strip()

cx = (df_cyl["x-coordinate"].min() + df_cyl["x-coordinate"].max()) / 2.0
cy = (df_cyl["y-coordinate"].min() + df_cyl["y-coordinate"].max()) / 2.0
R  = max((df_cyl["x-coordinate"].max() - df_cyl["x-coordinate"].min()) / 2.0,
         (df_cyl["y-coordinate"].max() - df_cyl["y-coordinate"].min()) / 2.0)
print(f"  Cylinder: center=({cx:.4f}, {cy:.4f})  R={R:.4f}")

XMIN = float(df_field["x-coordinate"].min())
XMAX = float(df_field["x-coordinate"].max())
YMIN = float(df_field["y-coordinate"].min())
YMAX = float(df_field["y-coordinate"].max())
print(f"  Domain : x in [{XMIN:.3f}, {XMAX:.3f}], y in [{YMIN:.3f}, {YMAX:.3f}]")

# Exclude points inside (or grazing) the cylinder
margin  = 0.01
r_field = np.sqrt((df_field["x-coordinate"] - cx)**2 +
                  (df_field["y-coordinate"] - cy)**2)
df_train = df_field[r_field > (R + margin)].reset_index(drop=True)

# ============================================================
# 2. Sensor placement: KMeans stratification
#    Centroids -> snap each to its nearest real CFD node
# ============================================================
coords_all = df_train[["x-coordinate", "y-coordinate"]].values
km = KMeans(n_clusters=N_SENSORS, n_init=10, random_state=SEED).fit(coords_all)
_, idx = cKDTree(coords_all).query(km.cluster_centers_, k=1)
df_sensors = df_train.iloc[idx].drop_duplicates().reset_index(drop=True)
print(f"  Sensors: {len(df_sensors)} (KMeans-stratified)")

x_np = df_sensors["x-coordinate"].values[:, None].astype(np.float32)
y_np = df_sensors["y-coordinate"].values[:, None].astype(np.float32)
u_np = df_sensors["x-velocity"].values[:, None].astype(np.float32)
v_np = df_sensors["y-velocity"].values[:, None].astype(np.float32)

# ============================================================
# 3. Network: stream-function PINN
# ============================================================
def norm_x(x): return 2.0 * (x - XMIN) / (XMAX - XMIN) - 1.0
def norm_y(y): return 2.0 * (y - YMIN) / (YMAX - YMIN) - 1.0

class StreamPINN(nn.Module):
    """Outputs (psi, p). Velocities derived by autograd: continuity is exact."""
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        layers = [2, *hidden, 2]
        seq = []
        for i in range(len(layers) - 1):
            lin = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            seq.append(lin)
            if i < len(layers) - 2:
                seq.append(ACTIVATION())
        self.net = nn.Sequential(*seq)

    def psi_p(self, x, y):
        out = self.net(torch.cat([norm_x(x), norm_y(y)], dim=1))
        return out[:, 0:1], out[:, 1:2]    # psi, p

    def velocities(self, x, y):
        psi, p = self.psi_p(x, y)
        u =  torch.autograd.grad(psi, y, torch.ones_like(psi),
                                 create_graph=True)[0]
        v = -torch.autograd.grad(psi, x, torch.ones_like(psi),
                                 create_graph=True)[0]
        return u, v, p

model = StreamPINN().to(DEVICE)
print(f"  Params : {sum(p.numel() for p in model.parameters())}")

# ============================================================
# 4. Tensors
# ============================================================
x_t = torch.tensor(x_np, requires_grad=True)
y_t = torch.tensor(y_np, requires_grad=True)
u_t = torch.tensor(u_np)
v_t = torch.tensor(v_np)

# Cylinder-surface BC points (no-slip)
theta   = np.linspace(0.0, 2*np.pi, N_BC, endpoint=False, dtype=np.float32)
x_bc    = torch.tensor((cx + R*np.cos(theta))[:, None], requires_grad=True)
y_bc    = torch.tensor((cy + R*np.sin(theta))[:, None], requires_grad=True)

# ============================================================
# 5. Losses
# ============================================================
mse = nn.MSELoss()

def loss_data():
    u_p, v_p, _ = model.velocities(x_t, y_t)
    return mse(u_p, u_t) + mse(v_p, v_t)

def loss_bc():
    u_p, v_p, _ = model.velocities(x_bc, y_bc)
    z = torch.zeros_like(u_p)
    return mse(u_p, z) + mse(v_p, z)

def sample_collocation(n):
    """Uniform random in box, reject inside cylinder."""
    xc = np.random.uniform(XMIN, XMAX, (n, 1)).astype(np.float32)
    yc = np.random.uniform(YMIN, YMAX, (n, 1)).astype(np.float32)
    keep = ((xc - cx)**2 + (yc - cy)**2) > (R + margin)**2
    return xc[keep[:, 0]].reshape(-1, 1), yc[keep[:, 0]].reshape(-1, 1)

def ns_residuals(xc, yc):
    u, v, p = model.velocities(xc, yc)
    u_x = torch.autograd.grad(u, xc, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, yc, torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, xc, torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, yc, torch.ones_like(v), create_graph=True)[0]
    p_x = torch.autograd.grad(p, xc, torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, yc, torch.ones_like(p), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, xc, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, yc, torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, xc, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, yc, torch.ones_like(v_y), create_graph=True)[0]
    fx = u*u_x + v*u_y + p_x - NU*(u_xx + u_yy)
    fy = u*v_x + v*v_y + p_y - NU*(v_xx + v_yy)
    return fx, fy

def loss_physics_resampled():
    xc_np, yc_np = sample_collocation(N_COLLOC_ADAM)
    xc = torch.tensor(xc_np, requires_grad=True)
    yc = torch.tensor(yc_np, requires_grad=True)
    fx, fy = ns_residuals(xc, yc)
    return mse(fx, torch.zeros_like(fx)) + mse(fy, torch.zeros_like(fy))

# Fixed collocation set for L-BFGS (line search needs deterministic loss)
xc_fix_np, yc_fix_np = sample_collocation(N_COLLOC_LBFGS)
xc_fix = torch.tensor(xc_fix_np, requires_grad=True)
yc_fix = torch.tensor(yc_fix_np, requires_grad=True)
def loss_physics_fixed():
    fx, fy = ns_residuals(xc_fix, yc_fix)
    return mse(fx, torch.zeros_like(fx)) + mse(fy, torch.zeros_like(fy))

# ============================================================
# 6. Phase 1 — Adam with physics-weight ramp
# ============================================================
hist = {"total": [], "data": [], "phys": [], "bc": []}
opt   = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)

print("\nPhase 1 -- Adam")
t0 = time.time()
for ep in range(ADAM_EPOCHS):
    lam_p = min(1.0, 0.01 + 0.99 * ep / (0.6 * ADAM_EPOCHS))
    opt.zero_grad()
    Ld = loss_data()
    Lp = loss_physics_resampled()
    Lb = loss_bc()
    L  = Ld + lam_p * Lp + LAM_BC * Lb
    L.backward(); opt.step(); sched.step()

    hist["total"].append(L.item()); hist["data"].append(Ld.item())
    hist["phys"].append(Lp.item());  hist["bc"].append(Lb.item())

    if ep % 500 == 0 or ep == ADAM_EPOCHS - 1:
        print(f"  ep {ep:5d} | total {L.item():.3e} | "
              f"data {Ld.item():.3e} | phys {Lp.item():.3e} | "
              f"bc {Lb.item():.3e} | lam_p {lam_p:.3f}")
print(f"  Adam time: {time.time()-t0:.1f}s")

# ============================================================
# 7. Phase 2 — L-BFGS (fixed collocation)
# ============================================================
print("\nPhase 2 -- L-BFGS")
opt_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    max_iter=LBFGS_MAX_ITER, history_size=50,
    tolerance_grad=1e-7, tolerance_change=1e-9,
    line_search_fn="strong_wolfe",
)
def closure():
    opt_lbfgs.zero_grad()
    L = loss_data() + loss_physics_fixed() + LAM_BC * loss_bc()
    L.backward()
    return L
t0 = time.time()
opt_lbfgs.step(closure)
print(f"  L-BFGS time: {time.time()-t0:.1f}s")
print(f"  Final loss : {closure().item():.3e}")

# ============================================================
# 8. Evaluation grid + plots
# ============================================================
print("\nGenerating plots...")
nx, ny = 300, 150
XX, YY = np.meshgrid(np.linspace(XMIN, XMAX, nx),
                     np.linspace(YMIN, YMAX, ny))
X_flat = torch.tensor(XX.flatten()[:, None].astype(np.float32), requires_grad=True)
Y_flat = torch.tensor(YY.flatten()[:, None].astype(np.float32), requires_grad=True)

U_, V_, P_ = model.velocities(X_flat, Y_flat)
U = U_.detach().numpy().reshape(XX.shape)
V = V_.detach().numpy().reshape(XX.shape)
P = P_.detach().numpy().reshape(XX.shape)
M = np.sqrt(U**2 + V**2)
mask = (XX - cx)**2 + (YY - cy)**2 <= R**2
for arr in (M, U, V, P): arr[mask] = np.nan
P -= np.nanmean(P)   # pressure is defined up to a constant

# (a) velocity magnitude with sensors overlaid
fig, ax = plt.subplots(figsize=(12, 5))
cs = ax.contourf(XX, YY, M, levels=80, cmap="turbo")
plt.colorbar(cs, ax=ax, label="|U| [m/s]")
ax.scatter(x_np, y_np, c="white", s=10, edgecolor="black",
           lw=0.4, label=f"{len(df_sensors)} sensors")
ax.add_patch(plt.Circle((cx, cy), R, color="black", zorder=10))
ax.set_aspect("equal"); ax.legend(loc="upper right")
ax.set_title("Reconstructed velocity magnitude")
ax.set_xlabel("x"); ax.set_ylabel("y")
fig.tight_layout(); fig.savefig("reconstructed_velocity.png", dpi=200)

# (b) inferred pressure
fig, ax = plt.subplots(figsize=(12, 5))
cs = ax.contourf(XX, YY, P, levels=80, cmap="RdBu_r")
plt.colorbar(cs, ax=ax, label="p (relative)")
ax.add_patch(plt.Circle((cx, cy), R, color="black", zorder=10))
ax.set_aspect("equal")
ax.set_title("PINN-inferred pressure (no pressure data used)")
ax.set_xlabel("x"); ax.set_ylabel("y")
fig.tight_layout(); fig.savefig("reconstructed_pressure.png", dpi=200)

# (c) streamlines
fig, ax = plt.subplots(figsize=(12, 5))
ax.streamplot(XX, YY, U, V, color=M, cmap="turbo",
              density=2.0, linewidth=0.8)
ax.add_patch(plt.Circle((cx, cy), R, color="black", zorder=10))
ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX); ax.set_aspect("equal")
ax.set_title("Reconstructed streamlines")
ax.set_xlabel("x"); ax.set_ylabel("y")
fig.tight_layout(); fig.savefig("reconstructed_streamlines.png", dpi=200)

# (d) loss history
fig, ax = plt.subplots(figsize=(8, 5))
for k in ("total", "data", "phys", "bc"):
    ax.semilogy(hist[k], label=k, lw=1.2)
ax.set_xlabel("Adam epoch"); ax.set_ylabel("loss")
ax.set_title("Training loss"); ax.grid(True, which="both", alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig("loss_history.png", dpi=200)

# (e) error vs CFD ground truth (over the full filtered field)
xt = torch.tensor(df_train["x-coordinate"].values[:, None].astype(np.float32),
                  requires_grad=True)
yt = torch.tensor(df_train["y-coordinate"].values[:, None].astype(np.float32),
                  requires_grad=True)
u_pred, v_pred, _ = model.velocities(xt, yt)
u_pred = u_pred.detach().numpy().ravel()
v_pred = v_pred.detach().numpy().ravel()
u_true = df_train["x-velocity"].values
v_true = df_train["y-velocity"].values
err = np.sqrt((u_pred - u_true)**2 + (v_pred - v_true)**2)
rel_l2 = (np.linalg.norm(np.r_[u_pred-u_true, v_pred-v_true])
          / np.linalg.norm(np.r_[u_true, v_true]))
print(f"\n  MAE on full field      : {err.mean():.4e}")
print(f"  Relative L2 (u, v)     : {rel_l2:.4f}")

fig, ax = plt.subplots(figsize=(12, 5))
sc = ax.scatter(df_train["x-coordinate"], df_train["y-coordinate"],
                c=err, s=2, cmap="magma")
plt.colorbar(sc, ax=ax, label="|u_PINN - u_CFD|")
ax.add_patch(plt.Circle((cx, cy), R, color="black", zorder=10))
ax.set_aspect("equal")
ax.set_title("Pointwise velocity error vs. CFD ground truth")
ax.set_xlabel("x"); ax.set_ylabel("y")
fig.tight_layout(); fig.savefig("error_vs_truth.png", dpi=200)

print("\nDone.")
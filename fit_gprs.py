"""
fit_gprs.py
-----------
fit_gprs(drone_pts, cfd_fields, solver) -> GPRs

One call that handles:
  - getting the potential-flow prior at drone points
  - fitting the divergence-free velocity GPR on residuals
  - fitting the scalar pressure GPR
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from divergence_free_gpr import DivergenceFreeGPR


@dataclass
class GPRs:
    vel:   DivergenceFreeGPR
    pres:  GaussianProcessRegressor

    def predict(self, pts: np.ndarray, prior_vel: np.ndarray):
        """Return velocity (N,3) and pressure (N,) at pts."""
        pts = np.asarray(pts, dtype=float)
        vel  = self.vel.predict(pts, prior_vel)
        pres = self.pres.predict(pts)
        return vel, pres


def fit_gprs(drone_pts: np.ndarray, cfd_fields, solver,
             vel_restarts: int = 8, pres_restarts: int = 8,
             posterior_batch: int = 4_000) -> GPRs:
    """
    Fit velocity + pressure GPRs from drone-point CFD samples.

    Parameters
    ----------
    drone_pts    : (N, 3) sampling locations
    cfd_fields   : CFDFields  (from cfd_sampler)
    solver       : FLOWPanelSolver  (already started)
    vel_restarts : hyperparameter optimisation restarts for velocity GPR
    pres_restarts: hyperparameter optimisation restarts for pressure GPR
    posterior_batch : chunk size for prediction (memory cap)

    Returns
    -------
    GPRs  dataclass with .vel, .pres, and .predict(pts, prior_vel)
    """
    prior = solver.velocity(drone_pts, blank_interior=False)
    if np.isnan(prior).any():
        raise RuntimeError("NaNs in FLOWPanel prior at drone points.")

    vel_gpr = DivergenceFreeGPR(
        n_restarts=vel_restarts, posterior_batch=posterior_batch,
    ).fit(drone_pts, cfd_fields.velocity - prior)

    p_gpr = GaussianProcessRegressor(
        kernel=Matern(length_scale=[1,1,1], length_scale_bounds=(1e-2, 1e3), nu=2.5),
        normalize_y=True, n_restarts_optimizer=pres_restarts, random_state=0,
    ).fit(drone_pts, cfd_fields.pressure)

    print(f"  vel  ell={vel_gpr.ell_}  var={vel_gpr.var_:.4g}  noise={vel_gpr.noise_:.4g}")
    print(f"  pres {p_gpr.kernel_}")
    return GPRs(vel=vel_gpr, pres=p_gpr)
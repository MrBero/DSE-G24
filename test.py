"""
Standalone diagnostic for the GPR wake pipeline.

Run this BEFORE touching hyperparameters. It answers two questions:
  1. Are the CFD frame, the training points, and the test grid in the SAME place?
     (If not, it's an input/coordinate-frame bug — no tuning will fix it.)
  2. Does the CFD interpolator return the right velocities at the CFD's own points?
     (If not, build_cfd_sampler is broken / over-smoothing / frame-mismatched.)

Edit the CONFIG block to match your main() call, then:  python diagnose.py
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pandas as pd
import trimesh

from sampling import sample
from flowpanelwrapper import FLOWPanelSolver

# ----------------------------------------------------------------------------
# CONFIG  — copy these straight from your main()
# ----------------------------------------------------------------------------
STL_FILEPATH = "input_stls/Aerospecial_building4.stl"
CFD_FILEPATH = "inputs/csv_with_everything.pkl"
STL_SCALE    = 1.0 / 1000.0
STL_ROTATE   = None
V_INF        = (0.0, 12.0, 0.0)
BOUNDS_INPUT = np.array([[-100, 100], [75, 275], [0, 50]], dtype=float)  # or None
SAMPLE_METHOD = "cylinder"
SAMPLE_CONFIG = {"r_factor": 3, "h_factor": 1.5, "tilt_deg": 0,
                 "n_points": 90, "front_frac": 0.5, "front_half_angle_deg": 45}
NUM_SAMPLES = 120
# ----------------------------------------------------------------------------


def banner(msg):
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


def fmt(arr):
    return np.array2string(np.asarray(arr, dtype=float), precision=2, suppress_small=True)


def main():
    # ---- Mesh (same transforms as run_gpr) ----
    banner("MESH")
    stl_mesh = trimesh.load_mesh(STL_FILEPATH)
    if STL_SCALE != 1.0:
        stl_mesh.apply_scale(STL_SCALE)
    if STL_ROTATE is not None:
        c = stl_mesh.centroid
        R = trimesh.transformations.rotation_matrix(STL_ROTATE, [0, 0, 1], c)
        stl_mesh.apply_transform(R)
    V = np.asarray(stl_mesh.vertices)
    print("mesh bounds  min:", fmt(V.min(0)), " max:", fmt(V.max(0)))
    print("mesh extents    :", fmt(V.max(0) - V.min(0)))
    print("mesh centroid   :", fmt(stl_mesh.centroid))

    # ---- Raw CFD frame (read the pkl directly, before any sampling) ----
    banner("CFD PKL FRAME (raw)")
    cfd = pd.read_pickle(CFD_FILEPATH)
    cfd.columns = cfd.columns.str.strip()
    src = cfd[["x-coordinate", "y-coordinate", "z-coordinate"]].to_numpy(float)
    vel = cfd[["x-velocity", "y-velocity", "z-velocity"]].to_numpy(float)
    print("CFD coord min:", fmt(src.min(0)), " max:", fmt(src.max(0)))
    print("CFD n_points :", len(src))
    print("CFD |vel| min/mean/max:",
          f"{np.linalg.norm(vel,axis=1).min():.2f} / "
          f"{np.linalg.norm(vel,axis=1).mean():.2f} / "
          f"{np.linalg.norm(vel,axis=1).max():.2f}")
    print("CFD vel per-comp min:", fmt(vel.min(0)), " max:", fmt(vel.max(0)))

    # ---- Run the actual sampler ----
    banner("SAMPLER OUTPUT")
    ground_truth, cfd_bounds, sample_dat_shi = sample(
        CFD_FILEPATH, stl_mesh,
        sample_method=SAMPLE_METHOD, num_samples=NUM_SAMPLES,
        sample_config=SAMPLE_CONFIG, epsilon=0.02, use_signed_distance=True,
    )
    tc = ground_truth[["x-target", "y-target", "z-target"]].to_numpy()
    tv = ground_truth[["x-velocity", "y-velocity", "z-velocity"]].to_numpy()
    print(f"requested num_samples : {NUM_SAMPLES}  "
          f"(sampler config n_points/etc may differ)")
    print(f"returned training pts : {len(tc)}")
    if len(tc) < 0.5 * NUM_SAMPLES:
        print("  >>> WARNING: large point loss. Sampler is rejecting most points.")
    print("training coord min:", fmt(tc.min(0)), " max:", fmt(tc.max(0)))
    print("training |vel| min/mean/max:",
          f"{np.linalg.norm(tv,axis=1).min():.2f} / "
          f"{np.linalg.norm(tv,axis=1).mean():.2f} / "
          f"{np.linalg.norm(tv,axis=1).max():.2f}")

    # ---- Frame overlap check: CFD vs test grid ----
    banner("FRAME OVERLAP  (the big one)")
    test_bounds = BOUNDS_INPUT if BOUNDS_INPUT is not None else cfd_bounds
    print("CFD pkl bounds   :", fmt(cfd_bounds))
    print("test grid bounds :", fmt(test_bounds))
    print("training extent  :", fmt(np.stack([tc.min(0), tc.max(0)], 1)))
    # do the test bounds and the CFD frame overlap on every axis?
    ax = "xyz"
    ok = True
    for i in range(3):
        lo = max(cfd_bounds[i, 0], test_bounds[i, 0])
        hi = min(cfd_bounds[i, 1], test_bounds[i, 1])
        overlap = hi - lo
        full = test_bounds[i, 1] - test_bounds[i, 0]
        frac = overlap / full if full > 0 else 0.0
        flag = "OK" if frac > 0.5 else ">>> MISMATCH"
        if frac <= 0.5:
            ok = False
        print(f"  {ax[i]}: test∩cfd overlap = {overlap:8.2f}  "
              f"({frac*100:5.1f}% of test range)   {flag}")
    if not ok:
        print("\n  >>> The test grid samples the CFD OUTSIDE its data range on some axis.")
        print("  >>> This is a coordinate-frame / bounds_input bug. Fix this FIRST;")
        print("  >>> no hyperparameter change will help while frames disagree.")
    else:
        print("\n  Frames overlap. Coordinate frame is probably fine.")

    # ---- Are training points inside the CFD frame? ----
    inside = np.all((tc >= cfd_bounds[:, 0]) & (tc <= cfd_bounds[:, 1]), axis=1)
    print(f"\ntraining points inside CFD frame: {inside.sum()}/{len(tc)}")
    if inside.sum() < len(tc):
        print("  >>> Some training points are outside the CFD data — interp gives garbage/NaN there.")

    # ---- Interpolator sanity: query at the CFD's OWN points ----
    banner("INTERPOLATOR SANITY  (query at CFD's own points)")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(src), size=min(12, len(src)), replace=False)
    probe_xyz = src[idx]
    probe_want = np.hstack([vel[idx],
                            cfd["pressure"].to_numpy(float)[idx, None]])
    probe_got = np.asarray(sample_dat_shi(probe_xyz))
    err = np.abs(probe_got[:, :3] - probe_want[:, :3])
    rel = err / (np.abs(probe_want[:, :3]) + 1e-6)
    print("per-point max abs vel error vs stored value:")
    print(fmt(err.max(1)))
    print(f"mean abs vel error : {err.mean():.4f}  (should be ~0 at the CFD's own nodes)")
    print(f"max  abs vel error : {err.max():.4f}")
    if err.max() > 0.5:
        print("  >>> Interpolator does NOT reproduce the CFD at its own points.")
        print("  >>> build_cfd_sampler is over-smoothing or in a mismatched frame.")
    else:
        print("  Interpolator reproduces stored velocities well. Interp is fine.")

    # ---- Prior vs training (residual scale the GP will see) ----
    banner("PRIOR / RESIDUAL SCALE")
    solver = FLOWPanelSolver(stl_mesh, np.asarray(V_INF, float),
                             julia_script="FP.jl", julia_bin="julia", verbose=False)
    means_train = solver.velocity(tc, blank_interior=False).reshape(-1, 3)
    if np.isnan(means_train).any():
        print("  >>> NaNs in prior at training points!")
    resid = tv - means_train
    print("prior |vel| at train  min/mean/max:",
          f"{np.linalg.norm(means_train,axis=1).min():.2f} / "
          f"{np.linalg.norm(means_train,axis=1).mean():.2f} / "
          f"{np.linalg.norm(means_train,axis=1).max():.2f}")
    print("residual per-comp  mean:", fmt(resid.mean(0)),
          " std:", fmt(resid.std(0)))
    print(f"residual overall  mean={resid.mean():.4f}  std={resid.std():.4f}  "
          f"min={resid.min():.4f}  max={resid.max():.4f}")
    print(f"\nuse this std ({resid.std():.4f}) to sanity-check var/noise bounds:")
    print("  noise bound should be WELL BELOW this; var (signal) comparable to std^2.")

    banner("SUMMARY")
    print("1. Frames overlap?      ", "YES" if ok else "NO  <-- fix first")
    print("2. Training inside CFD? ", f"{inside.sum()}/{len(tc)}")
    print("3. Interp reproduces?   ",
          "YES" if err.max() <= 0.5 else "NO  <-- build_cfd_sampler bug")
    print(f"4. Residual std         {resid.std():.3f}  "
          f"(noise bound must be << this)")
    print(f"5. Point yield          {len(tc)}/{NUM_SAMPLES}")


if __name__ == "__main__":
    main()
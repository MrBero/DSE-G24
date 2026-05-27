import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.spatial
import scipy.interpolate
import jax as jax
import pandas as pd
import os

# setup
seed = 7
rnd.seed(seed)

# GROUND TRUTH SUBSAMPLING FUNCTION 
# Inputs:
# field_path -- path to the field data (csv) of the underlying CFD truth
# wall_path -- path to the wall coordinates (csv)
# samples (CSV file path OR np array) -- (N, 3) array of points within the domain to be sampled.
# method (string) -- "random", "CSV", or "array".
# epsilon -- length (in meters) of margin of sampling from building walls
# num_samples -- number of random points to generate if method is "random"

def sample(field_path, wall_path, samples=None, method="CSV", epsilon=0.02, num_samples=2000):

    # Load field ground truth
    cfd_df = pd.read_csv(field_path)
    cfd_df.columns = cfd_df.columns.str.strip()
    source_coords = cfd_df[['x-coordinate', 'y-coordinate', 'z-coordinate']].values
    source_values = cfd_df[['x-velocity', 'y-velocity', 'z-velocity', 'pressure']].values

    # Bounds of the domain; used later for random sampling and validity checks
    x_min, x_max = source_coords[:, 0].min(), source_coords[:, 0].max()
    y_min, y_max = source_coords[:, 1].min(), source_coords[:, 1].max()
    z_min, z_max = source_coords[:, 2].min(), source_coords[:, 2].max()

    # Load wall data and construct tree for optimized distance checking
    wall_df = pd.read_csv(wall_path)
    wall_df.columns = wall_df.columns.str.strip()
    
    if 'x-coordinate' in wall_df.columns:
        wall_points = wall_df[['x-coordinate', 'y-coordinate', 'z-coordinate']].values
    else:
        wall_points = wall_df.iloc[:, :3].values 

    wall_tree = sp.spatial.cKDTree(wall_points)

    # Resolve Target Points based on Method
    if method == "random":
        valid_points_list = []
        collected = 0
        
        # Keep generating in batches until we hit the requested num_samples
        while collected < num_samples:
            batch_size = num_samples - collected
            x_rand = rnd.uniform(x_min, x_max, batch_size)
            y_rand = rnd.uniform(y_min, y_max, batch_size)
            z_rand = rnd.uniform(z_min, z_max, batch_size)
            batch_points = np.column_stack((x_rand, y_rand, z_rand))
            
            # Distance check against the KD-Tree
            distances, _ = wall_tree.query(batch_points)
            safe_points = batch_points[distances > epsilon]
            
            if len(safe_points) > 0:
                valid_points_list.append(safe_points)
                collected += len(safe_points)
                
        # Stack all the safe batches and trim perfectly to num_samples
        valid_points = np.vstack(valid_points_list)[:num_samples]

    else:
        # Load targets based on CSV or array
        if method == "CSV":
            target_df = pd.read_csv(samples)
            target_df.columns = target_df.columns.str.strip()
            
            if 'x-coordinate' in target_df.columns:
                target_points = target_df[['x-coordinate', 'y-coordinate', 'z-coordinate']].values
            else:
                target_points = target_df.iloc[:, :3].values
                
        elif method == "array":
            target_points = np.array(samples)
        else:
            raise ValueError("Method must be 'CSV', 'array', or 'random'.")
        
        # protect against 1d arrays (1 point)
        if target_points.ndim == 1:
            target_points = target_points.reshape(1, -1)

        # mask out points outside of the CFD ground truth region bounds
        bounds_mask = (
            (target_points[:, 0] >= x_min) & (target_points[:, 0] <= x_max) &
            (target_points[:, 1] >= y_min) & (target_points[:, 1] <= y_max) &
            (target_points[:, 2] >= z_min) & (target_points[:, 2] <= z_max))
        valid_points = target_points[bounds_mask]

        # mask out points too close to the wall
        if len(valid_points) > 0:
            distances, _ = wall_tree.query(valid_points)
            valid_points = valid_points[distances > epsilon]
                
        if len(valid_points) == 0:
            raise ValueError("No points left to sample! All points were out of bounds or too close to a wall.")

    # Shared Interpolation Block (Applies to all methods)
    interpolated_values = sp.interpolate.griddata(
        points=source_coords,
        values=source_values,
        xi=valid_points,
        method='linear',
        fill_value=np.nan 
    )

    # Package Data and Return
    columns = ['x-target', 'y-target', 'z-target', 'x-velocity', 'y-velocity', 'z-velocity', 'pressure']
    results_df = pd.DataFrame(
        np.hstack((valid_points, interpolated_values)), 
        columns=columns
    )
        
    return results_df



import numpy as np
import numpy.random as rnd
import scipy as sp
import pandas as pd

seed = 7
rnd.seed(seed)


def sample(field_path, stl_mesh, samples=None, method="CSV",
           epsilon=0.02, num_samples=100):
    cfd_df = pd.read_csv(field_path)
    cfd_df.columns = cfd_df.columns.str.strip()
    source_coords = cfd_df[['x-coordinate', 'y-coordinate', 'z-coordinate']].values
    source_values = cfd_df[['x-velocity', 'y-velocity', 'z-velocity', 'pressure']].values

    x_min, x_max = source_coords[:, 0].min(), source_coords[:, 0].max()
    y_min, y_max = source_coords[:, 1].min(), source_coords[:, 1].max()
    z_min, z_max = source_coords[:, 2].min(), source_coords[:, 2].max()
    bounds = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])

    wall_points = np.asarray(stl_mesh.vertices)
    wall_tree = sp.spatial.cKDTree(wall_points)

    if method == "random":
        valid_points_list = []
        collected = 0
        while collected < num_samples:
            batch_size = num_samples - collected
            x_rand = rnd.uniform(x_min, x_max, batch_size)
            y_rand = rnd.uniform(y_min, y_max, batch_size)
            z_rand = rnd.uniform(z_min, z_max, batch_size)
            batch_points = np.column_stack((x_rand, y_rand, z_rand))
            distances, _ = wall_tree.query(batch_points)
            safe_points = batch_points[distances > epsilon]
            if len(safe_points) > 0:
                valid_points_list.append(safe_points)
                collected += len(safe_points)
        valid_points = np.vstack(valid_points_list)[:num_samples]
    else:
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

        if target_points.ndim == 1:
            target_points = target_points.reshape(1, -1)

        bounds_mask = (
            (target_points[:, 0] >= x_min) & (target_points[:, 0] <= x_max) &
            (target_points[:, 1] >= y_min) & (target_points[:, 1] <= y_max) &
            (target_points[:, 2] >= z_min) & (target_points[:, 2] <= z_max))
        valid_points = target_points[bounds_mask]

        if len(valid_points) > 0:
            distances, _ = wall_tree.query(valid_points)
            valid_points = valid_points[distances > epsilon]
        if len(valid_points) == 0:
            raise ValueError("No points left to sample! All out of bounds or too close to a wall.")

    interpolated_values = sp.interpolate.griddata(
        points=source_coords, values=source_values,
        xi=valid_points, method='linear', fill_value=np.nan)

    columns = ['x-target', 'y-target', 'z-target',
               'x-velocity', 'y-velocity', 'z-velocity', 'pressure']
    results_df = pd.DataFrame(np.hstack((valid_points, interpolated_values)), columns=columns)
    before = len(results_df)
    results_df = results_df.dropna().reset_index(drop=True)
    dropped = before - len(results_df)
    return results_df, bounds
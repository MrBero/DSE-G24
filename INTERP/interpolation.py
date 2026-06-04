import numpy as np
import pandas as pd
import pyvista as pv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PKL_PATH  = r"inputs/csv_with_everything.pkl"
N_POINTS  = 4      # number of nearest neighbours used for interpolation
SHARPNESS = 2.0    # IDW falloff sharpness


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------
def build_cfd_sampler(df, n_points: int = 8, sharpness: float = 2.0):
    """
    Load a CFD pickle and return a sampler function.

    Parameters
    ----------
    pkl_path  : path to .pkl with columns x/y/z-coordinate, x/y/z-velocity, pressure
    n_points  : number of nearest neighbours used for IDW interpolation
    sharpness : IDW falloff sharpness
    """

    cloud = pv.PolyData(df[['x-coordinate', 'y-coordinate', 'z-coordinate']].values)
    cloud['x-velocity'] = df['x-velocity'].values
    cloud['y-velocity'] = df['y-velocity'].values
    cloud['z-velocity'] = df['z-velocity'].values
    cloud['pressure']   = df['pressure'].values

    def sample_dat_shi(points_or_x, y=None, z=None) -> dict:
        """
        Interpolate velocity and pressure at arbitrary query points.

        Parameters
        ----------
        points_or_x : (N, 3) array-like  e.g. [[0,0,0], [1,2,3]]
                      or 1-D x array when y and z are passed separately
        y, z        : 1-D arrays, required when passing coordinates separately

        Returns
        -------
        dict with:
            'velocity' : np.ndarray (N, 3)  — [vx, vy, vz]
            'pressure' : np.ndarray (N,)
        """
        pts = np.asarray(points_or_x, dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 3:
            query_pts = pts
        else:
            query_pts = np.column_stack([pts, np.asarray(y, float), np.asarray(z, float)])

        result = pv.PolyData(query_pts).interpolate(
            cloud,
            n_points=n_points,
            sharpness=sharpness,
            strategy='null_value',
        )
        return np.stack([result['x-velocity'],
                        result['y-velocity'],
                        result['z-velocity'],
                        result['pressure']], axis=1)

    return sample_dat_shi


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    sample = build_cfd_sampler(PKL_PATH, n_points=N_POINTS, sharpness=SHARPNESS)

    out = sample([[1, 2, 1], [5, 2, 1], [-50, 2, 1],[-50, 2, 1],[-50, 2, 1]])
    print("velocity :\n", out['velocity'])
    print("pressure :\n", out['pressure'])
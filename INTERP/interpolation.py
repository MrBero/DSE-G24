import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import joblib
import hashlib
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PKL_PATH   = r"inputs/csv_with_everything.pkl"
CACHE_PATH = r"inputs/cfd_sampler_cache.joblib"
N_POINTS   = 4
SHARPNESS  = 2.0


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------
def _df_hash(df: pd.DataFrame) -> str:
	"""Quick hash of the dataframe contents to detect if source data changed."""
	return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def build_cfd_sampler(df: pd.DataFrame, n_points: int = 8, sharpness: float = 2.0,
                      cache_path: str = None):
	"""
	Build (or load from cache) a fast IDW sampler from a CFD dataframe.

	Parameters
	----------
	df         : DataFrame with columns x/y/z-coordinate, x/y/z-velocity, pressure
	n_points   : number of nearest neighbours used for IDW interpolation
	sharpness  : IDW distance-weight exponent (higher = sharper falloff)
	cache_path : optional path to cache the cKDTree and fields to disk
	"""
	coords = df[['x-coordinate', 'y-coordinate', 'z-coordinate']].to_numpy(dtype=float)
	fields = df[['x-velocity', 'y-velocity', 'z-velocity', 'pressure']].to_numpy(dtype=float)

	# Try loading from cache
	if cache_path and os.path.exists(cache_path):
		print("Loading sampler from cache...")
		cache = joblib.load(cache_path)
		if cache['hash'] == _df_hash(df):
			tree   = cache['tree']
			fields = cache['fields']
			print("Cache hit -- skipping cKDTree build.")
		else:
			print("Cache miss (data changed) -- rebuilding...")
			cache = None
	else:
		cache = None

	# Build fresh if no valid cache
	if cache is None:
		print("Building cKDTree...")
		tree = cKDTree(coords)
		if cache_path:
			joblib.dump({'hash': _df_hash(df), 'tree': tree, 'fields': fields}, cache_path)
			print(f"Cache saved to {cache_path}")

	# Warm up process pool
	tree.query(coords[:1], k=n_points, workers=-1)

	def sample(points_or_x, y=None, z=None) -> np.ndarray:
		"""
		Interpolate velocity and pressure at arbitrary query points via IDW.

		Parameters
		----------
		points_or_x : (N, 3) array-like  e.g. [[0,0,0], [1,2,3]]
		              or 1-D x array when y and z are passed separately
		y, z        : 1-D arrays, required when passing coordinates separately

		Returns
		-------
		np.ndarray of shape (N, 4) -- columns: [vx, vy, vz, pressure]
		"""
		pts = np.asarray(points_or_x, dtype=float)
		if pts.ndim == 1 and y is not None:
			query_pts = np.column_stack([pts, np.asarray(y, float), np.asarray(z, float)])
		else:
			query_pts = pts if pts.ndim == 2 else pts.reshape(1, 3)

		dists, idx = tree.query(query_pts, k=n_points, workers=-1)

		exact = dists[:, 0] == 0.0
		weights = 1.0 / np.where(dists == 0.0, 1.0, dists) ** sharpness

		weights[exact] = 0.0
		weights[exact, 0] = 1.0
		weights /= weights.sum(axis=1, keepdims=True)

		return np.einsum('qk,qkf->qf', weights, fields[idx])

	return sample


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------
if __name__ == '__main__':
	df = pd.read_pickle(PKL_PATH)
	sample = build_cfd_sampler(df, n_points=N_POINTS, sharpness=SHARPNESS, cache_path=CACHE_PATH)

	out = sample([[1, 2, 1], [5, 2, 1], [-50, 2, 1]])
	print("result:\n", out)
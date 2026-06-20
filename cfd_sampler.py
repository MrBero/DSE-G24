from __future__ import annotations
import os
from dataclasses import dataclass

import numpy as np
import joblib
from scipy.spatial import cKDTree


@dataclass
class CFDFields:
	"""
	Readable container for interpolated CFD fields.

	All arrays are shape (N, ...) and correspond row-for-row to the
	query points passed into the sampler.
	"""
	velocity: np.ndarray         # (N, 3) -- [vx, vy, vz]
	pressure: np.ndarray         # (N,)
	turb_kin_energy: np.ndarray  # (N,)  -- k
	turb_visc: np.ndarray        # (N,)  -- mu_t

	def stacked(self) -> np.ndarray:
		"""Return the old-style (N, 6) array: [vx, vy, vz, p, k, mu_t]."""
		return np.column_stack([
			self.velocity, self.pressure, self.turb_kin_energy, self.turb_visc,
		])


def build_cfd_sampler(df, n_points: int = 8, sharpness: float = 2.0,
					  cache_path: str = None):
	"""
	Build (or load from cache) a fast IDW sampler from a CFD dataframe.
	Parameters
	----------
	df         : DataFrame with columns x/y/z-coordinate, x/y/z-velocity,
				 pressure, turb-kinetic-energy, viscosity-turb
	n_points   : number of nearest neighbours used for IDW interpolation
	sharpness  : IDW distance-weight exponent (higher = sharper falloff)
	cache_path : optional path to cache the cKDTree and fields to disk
	"""
	coords = df[['x-coordinate', 'y-coordinate', 'z-coordinate']].to_numpy(dtype=float)
	fields = df[[
		'x-velocity', 'y-velocity', 'z-velocity',
		'pressure',
		'turb-kinetic-energy',
		'viscosity-turb',
	]].to_numpy(dtype=float)

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

	def sample(points_or_x, y=None, z=None) -> CFDFields:
		"""
		Interpolate velocity, pressure, turbulent kinetic energy and
		turbulent viscosity at arbitrary query points via IDW.

		Parameters
		----------
		points_or_x : (N, 3) array-like  e.g. [[0,0,0], [1,2,3]]
					  or 1-D x array when y and z are passed separately
		y, z        : 1-D arrays, required when passing coordinates separately

		Returns
		-------
		CFDFields  -- dataclass with .velocity (N,3), .pressure (N,),
					  .turb_kin_energy (N,), .turb_visc (N,)
		"""
		pts = np.asarray(points_or_x, dtype=float)
		if pts.ndim == 1 and y is not None:
			query_pts = np.column_stack([pts, np.asarray(y, float), np.asarray(z, float)])
		else:
			query_pts = pts if pts.ndim == 2 else pts.reshape(1, 3)

		dists, idx = tree.query(query_pts, k=n_points, workers=-1)
		if n_points == 1:
			dists = dists[:, None]
			idx = idx[:, None]

		exact = dists[:, 0] == 0.0
		weights = 1.0 / np.where(dists == 0.0, 1.0, dists) ** sharpness
		weights[exact] = 0.0
		weights[exact, 0] = 1.0
		weights /= weights.sum(axis=1, keepdims=True)

		out = np.einsum('qk,qkf->qf', weights, fields[idx])

		return CFDFields(
			velocity=out[:, 0:3],
			pressure=out[:, 3],
			turb_kin_energy=out[:, 4],
			turb_visc=out[:, 5],
		)

	return sample
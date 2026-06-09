"""
divergence_free_gpr.py
----------------------
Self-contained divergence-free vector-valued GPR (Gaussian Process Regression)
using a Helmholtz-derived Matern-5/2 kernel.

Usage
-----
    from divergence_free_gpr import DivergenceFreeGPR

    gpr = DivergenceFreeGPR(n_restarts=8, posterior_batch=4000).fit(
        training_coords, residuals
    )
    posterior_vels = gpr.predict(test_points, prior_means)
    posterior_vars = gpr.predict_var(test_points)
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import scipy.optimize as spo
import scipy.stats.qmc as qmc

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

jax.config.update("jax_enable_x64", True)


# =============================================================================
# Module-level kernel functions
# These stay at module level because jax.jit on instance methods doesn't work
# cleanly -- jit traces all arguments including self, which breaks caching.
# =============================================================================

# Kernelissimo Kernelismus
def matern52_np(v1, v2, ell, var):
	"""Scalar Matérn-5/2 covariance. v1,v2:(3,)  ell:(3,)  var:scalar."""
	diff = v2 - v1
	r = jnp.sqrt(
		(diff[0] / ell[0]) ** 2
		+ (diff[1] / ell[1]) ** 2
		+ (diff[2] / ell[2]) ** 2
		+ 1e-8 # so we don't run into strange stuff at r=0
	)
	return var * (1.0 + jnp.sqrt(5.0) * r + (5.0 / 3.0) * r ** 2) * jnp.exp(-jnp.sqrt(5.0) * r)

# full derivation of 3x3 k0 matrix in the final report
# compile just in time for speed
@jax.jit
def Hemholtz_K0(V1, V2, ell, var):
	"""3x3 divergence-free vector kernel block from Hessian of scalar Matérn-5/2."""
	H = jax.hessian(matern52_np)(V1, V2, ell, var)
	return jnp.array(
		[
			[-H[1, 1] - H[2, 2], H[0, 1], H[0, 2]],
			[H[1, 0], -H[2, 2] - H[0, 0], H[1, 2]],
			[H[2, 0], H[2, 1], -H[0, 0] - H[1, 1]],
		]
	)

# assemble the covariance matrix from K0 matrices
def assemble_dat_shi(points_1, points_2, ell, var, noise_std=0.0, jitter=0.0):
	"""
	Block vector-valued covariance, point-major ordering:
		[u_x(p0), u_y(p0), u_z(p0), u_x(p1), ...]
	Returns (3*len(points_1), 3*len(points_2)).
	"""
	points_1 = jnp.asarray(points_1)
	points_2 = jnp.asarray(points_2)
	n_1 = points_1.shape[0]
	n_2 = points_2.shape[0]

	blocks = jax.vmap(
		lambda a: jax.vmap(lambda b: Hemholtz_K0(a, b, ell, var))(points_2)
	)(points_1)

	result_matrix = jnp.transpose(blocks, (0, 2, 1, 3)).reshape(n_1 * 3, n_2 * 3)

	if n_1 == n_2:
		if noise_std > 0.0:
			result_matrix = result_matrix + (noise_std ** 2 + jitter) * jnp.eye(n_1 * 3)
		elif jitter > 0.0:
			result_matrix = result_matrix + jitter * jnp.eye(n_1 * 3)

	return result_matrix


# =============================================================================
# DivergenceFreeGPR
# =============================================================================

class DivergenceFreeGPR:
	"""
	Divergence-free vector-valued GPR using a Helmholtz-derived Matern-5/2 kernel.

	Mirrors the sklearn GPR interface: construct, .fit(), .predict(), .predict_var().

	Parameters
	----------
	n_restarts : int
		Number of L-BFGS-B (Limited-memory Broyden-Fletcher-Goldfarb-Shanno
		Bounded) restarts for hyperparameter optimisation.
	jitter : float
		Diagonal jitter added during hyperparameter fitting to keep K (the
		covariance matrix) positive definite in bad regions of parameter space.
	posterior_batch : int
		Chunk size for batched posterior evaluation to bound peak memory.
	seed : int
		RNG (Random Number Generator) seed for Latin Hypercube start points.

	Attributes (set after .fit())
	------------------------------
	ell_   : (3,) array   -- fitted anisotropic length scales
	var_   : float        -- fitted signal variance
	noise_ : float        -- fitted noise standard deviation
	nll_   : float        -- negative log likelihood at optimum
	"""

	def __init__(self, n_restarts=8, jitter=1e-6, posterior_batch=4000, seed=0):
		self.n_restarts = n_restarts
		self.jitter = jitter
		self.posterior_batch = posterior_batch
		self.seed = seed

		# set after .fit()
		self.ell_ = None
		self.var_ = None
		self.noise_ = None
		self.nll_ = None
		self.alpha_ = None
		self._c = None
		self._low = None
		self._training_coords = None

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------

	def fit(self, training_coords, residuals):
		"""
		Fit hyperparameters and solve for alpha on the given residuals.

		Parameters
		----------
		training_coords : (n, 3) array
		residuals       : (3n, 1) or (n, 3) array  -- y minus prior mean

		Returns
		-------
		self
		"""
		training_coords = jnp.asarray(training_coords)
		residuals = jnp.asarray(residuals).reshape(-1, 1)

		fit = self._fit_hyperparams(training_coords, residuals)
		self.ell_   = jnp.asarray(fit["ell"])
		self.var_   = float(fit["var"])
		self.noise_ = float(fit["noise"])
		self.nll_   = float(fit["nll"])

		K = assemble_dat_shi(
			training_coords, training_coords,
			self.ell_, self.var_,
			noise_std=self.noise_, jitter=1e-8
		)
		self._c, self._low = cho_factor(K)
		self.alpha_ = cho_solve((self._c, self._low), residuals) # inverted K matrix times residuals is alpha
		self._training_coords = training_coords
		return self

	# posterior mean calculation, originally computed as one. We ran into memory issues... so now it is batched in sets of =batch points.
	def predict(self, test_points, means_tests, progress_every=0):
		"""Stream the GP posterior mean over chunks of test points. Returns (m, 3)."""
		self._check_fitted()
		test_points = np.asarray(test_points)
		means_tests = np.asarray(means_tests).reshape(-1, 3)
		n_test = test_points.shape[0]
		out = np.empty((n_test, 3), dtype=float)
		n_chunks = (n_test + self.posterior_batch - 1) // self.posterior_batch

		for ci, i in enumerate(range(0, n_test, self.posterior_batch)):
			tp = test_points[i:i + self.posterior_batch]
			# cross-covariance K(X_*, X)
			ks = assemble_dat_shi(tp, self._training_coords, self.ell_, self.var_, noise_std=0.0, jitter=0.0)
			# now do
			contrib = np.array(ks @ self.alpha_).reshape(-1, 3) # calculate prediction at the point by doing k(x_*, x) @ (K + sigma^2)^-1 (y-y_mean)
			out[i:i + self.posterior_batch] = means_tests[i:i + self.posterior_batch] + contrib # write output by adding mean to deviation prediction
			if progress_every and (ci % progress_every == 0 or ci == n_chunks - 1):
				print(f"    posterior vels chunk {ci + 1}/{n_chunks}", flush=True)

		return out

	# calculate the posterior variance in batches... again, because the matrix was taking up 23 gb of ram.
	# The equation we're using here for variance is: Sigma_* = K(X_*, X_*) - K(X_*, X) * (K(X, X) + noise^2 I)^-1 * K(X_*, X)^T
	def predict_var(self, test_points, progress_every=0):
		"""Stream the GP posterior variance diagonal over chunks. Returns (m, 3)."""
		self._check_fitted()
		test_points = np.asarray(test_points)
		n_test = test_points.shape[0]
		out = np.empty((n_test * 3,), dtype=float)
		n_chunks = (n_test + self.posterior_batch - 1) // self.posterior_batch

		for ci, i in enumerate(range(0, n_test, self.posterior_batch)):
			chunk = test_points[i:i + self.posterior_batch]
			m = chunk.shape[0]

			K_tt = assemble_dat_shi(chunk, chunk, self.ell_, self.var_)              # (3m, 3m)
			k_tc = assemble_dat_shi(chunk, self._training_coords, self.ell_, self.var_)    # (3m, 3n)
			beta_chunk = cho_solve((self._c, self._low), jnp.asarray(k_tc.T))        # (3n, 3m)
			diag = jnp.diag(K_tt - k_tc @ beta_chunk)                    # (3m,)

			out[i * 3:(i + m) * 3] = np.asarray(diag)
			if progress_every and (ci % progress_every == 0 or ci == n_chunks - 1):
				print(f"    posterior vars chunk {ci + 1}/{n_chunks}", flush=True)

		return out.reshape(-1, 3)

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	# Hyperparam fitting, the fun stuff
	def _fit_hyperparams(self, X, y):
		n = X.shape[0] # number of points
		# lower and upper bound in log space
		lo = np.log([5.0,  5.0,  5.0,  1e-1, 1e-2])
		hi = np.log([150.0, 150.0, 150.0, 1e3,  2e0])

		@jax.jit
		# find negative log likelihood
		def nll(log_theta):
			ell = jnp.exp(log_theta[:3]) # exponentiate length scales to convert back from log space
			var = jnp.exp(log_theta[3]) # same with variance
			noise = jnp.exp(log_theta[4]) # same with noise
			# assemble K matrix (as above)
			blocks = jax.vmap(lambda a: jax.vmap(lambda b: Hemholtz_K0(a, b, ell, var))(X))(X)
			K = jnp.transpose(blocks, (0, 2, 1, 3)).reshape(3 * n, 3 * n)
			K += (noise ** 2 + self.jitter) * jnp.eye(3 * n) # add noise and jitter on the diagonal

			c, low = cho_factor(K) # apply cholesky decomp (K = LL^T)
			alpha = cho_solve((c, low), y) # solve for weights
			logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(c))) # calculate the determinant by using the fact that it's the sum of diagonal entries of c squared.
			# return negative log likelihood, this is pretty standard
			return 0.5 * (y.T @ alpha)[0, 0] + 0.5 * logdet + 0.5 * (3 * n) * jnp.log(2 * jnp.pi)

		# find value and gradient using autodiff for the objective function
		nll_vg = jax.jit(jax.value_and_grad(nll))

		# objective function, helper for the spo optimizer later
		def objective(log_theta):
			val, grad = nll_vg(jnp.asarray(log_theta))
			return float(val), np.asarray(grad, dtype=float)

		# starting points spaced in the lo-high space in a latin hypercube to cover the space
		starts = lo + qmc.LatinHypercube(d=5, seed=self.seed).random(self.n_restarts) * (hi - lo)

		# optimizer loop
		best = None
		for t0 in starts:
			try:
				# optimizer function for spo using the L-BFGS-B method, using autodiff from objective, between higher and lower bound, max 200 steps.
				res = spo.minimize(
					objective, t0, method="L-BFGS-B", jac=True,
					bounds=list(zip(lo, hi)), options={"maxiter": 200}
				)
				if best is None or res.fun < best.fun:
					best = res
			except Exception as e:
				print(f"Restart failed")

		if best is None:
			raise RuntimeError("All optimizer restarts crashed. The math broke.")

		# recover theta from log space
		theta = np.exp(best.x)
		# return hyperparams
		return {"ell": theta[:3], "var": float(theta[3]), "noise": float(theta[4]),
				"nll": float(best.fun)}

	def _check_fitted(self):
		if self.alpha_ is None:
			raise RuntimeError("Call .fit() before predicting.")
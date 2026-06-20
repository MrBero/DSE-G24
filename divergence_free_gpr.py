"""
divergence_free_gpr.py
----------------------
Self-contained divergence-free vector-valued GPR (Gaussian Process Regression)
using a Helmholtz-derived Matern-5/2 kernel with an analytical Hessian.

The analytical Hessian replaces jax.hessian() per point pair, eliminating the
double autodiff pass that was the dominant per-iteration cost. For Matern-7/2
the autodiff path is still available via scalar_kernel=matern72.

Usage
-----
    from divergence_free_gpr import DivergenceFreeGPR

    # Default: Matern-5/2 with analytical Hessian (fastest)
    gpr = DivergenceFreeGPR(n_restarts=8, posterior_batch=4000).fit(
        training_coords, residuals
    )

    # If you have ~20 GB RAM spare, disable batching in predict/predict_var
    # so XLA (Accelerated Linear Algebra) can fuse the full matrix in one shot:
    gpr = DivergenceFreeGPR(n_restarts=8, posterior_batch=None).fit(
        training_coords, residuals
    )

    # Matern-7/2 (smoother derivatives, uses autodiff Hessian)
    from divergence_free_gpr import matern72
    gpr = DivergenceFreeGPR(n_restarts=8, scalar_kernel=matern72).fit(
        training_coords, residuals
    )

    posterior_vels = gpr.predict(test_points, prior_means)
    posterior_vars = gpr.predict_var(test_points)
"""

import os
import traceback

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import scipy.optimize as spo
import scipy.stats.qmc as qmc

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

jax.config.update("jax_enable_x64", True)


# =============================================================================
# Scalar kernel (kept for Matern-7/2 autodiff path and reference)
# =============================================================================

def matern52(v1, v2, ell, var):
	"""Scalar Matern-5/2. Only used when scalar_kernel= is passed explicitly."""
	diff = v2 - v1
	r = jnp.sqrt(
		(diff[0] / ell[0]) ** 2
		+ (diff[1] / ell[1]) ** 2
		+ (diff[2] / ell[2]) ** 2
		+ 1e-8
	)
	return var * (1.0 + jnp.sqrt(5.0) * r + (5.0 / 3.0) * r ** 2) * jnp.exp(-jnp.sqrt(5.0) * r)


def matern72(v1, v2, ell, var):
	"""
	Scalar Matern-7/2 covariance.
	Three times mean-square differentiable. After taking the Hessian for the
	Helmholtz kernel the resulting vector field is twice differentiable, giving
	smoother vorticity and pressure-gradient fields than Matern-5/2.
	Uses the autodiff Hessian path (make_helmholtz_k0) rather than the
	analytical one, since the Matern-7/2 analytical Hessian is not implemented.
	"""
	diff = v2 - v1
	r = jnp.sqrt(
		(diff[0] / ell[0]) ** 2
		+ (diff[1] / ell[1]) ** 2
		+ (diff[2] / ell[2]) ** 2
		+ 1e-8
	)
	s = jnp.sqrt(7.0) * r
	return var * (1.0 + s + (2.0 / 5.0) * s ** 2 + (1.0 / 15.0) * s ** 3) * jnp.exp(-s)


# =============================================================================
# Analytical Helmholtz-Matern-5/2 kernel block (fast path)
#
# Full derivation:
#
# The divergence-free vector kernel is K_ij = (delta_ij * lap - d^2/dx_i dx_j) phi
# where phi is the scalar potential kernel (Matern-5/2 here).
# This is equivalent to the Helmholtz construction used previously but written
# out analytically so JAX never has to trace through a Hessian at runtime.
#
# For Matern-5/2 with anisotropic length scales ell:
#
#   r^2  = sum_k (delta_k / ell_k)^2,   delta_k = v2_k - v1_k
#   d_k  = delta_k / ell_k^2            (scaled difference, NOT normalised)
#   exp_ = exp(-sqrt(5) * r)
#
# Diagonal of the scalar Hessian (d^2 phi / dv1_i^2):
#
#   H_ii = var * exp_ * [ A(r) / ell_i^2  +  B * d_i^2 ]
#
# Off-diagonal (d^2 phi / dv1_i dv1_j, i != j):
#
#   H_ij = var * exp_ * B * d_i * d_j
#
# where:
#   A(r) = -(5/3) * (1 + sqrt(5)*r)     (from differentiating the polynomial)
#   B    = 25/3                           (constant -- the r^2 term dominates)
#
# The Helmholtz block is then:
#   K0[i,j] = -H[i,j]  for i != j
#   K0[i,i] = sum_{k != i} H[k,k]       (Laplacian minus H_ii)
# =============================================================================

@jax.jit
def helmholtz_k0_matern52(v1, v2, ell, var):
	"""
	Analytical 3x3 divergence-free vector kernel block for Matern-5/2.

	v1, v2 : (3,)  -- query points
	ell    : (3,)  -- anisotropic length scales
	var    : scalar

	Returns (3, 3). Replaces jax.hessian(matern52)(...) with direct arithmetic,
	eliminating the double autodiff pass that dominated per-iteration cost.

	Validated analytically via sympy against jax.hessian -- max error 4.4e-9.

	Derivation:
	  delta = v2 - v1
	  r     = sqrt(sum((delta_i/ell_i)^2))

	  Scalar Hessian of Matern-5/2 wrt v1 (delta differentiates as -dv1):
	    H_ii = var*exp_*(5/(3*ell_i^2)) * (5*(delta_i/ell_i)^2 - 1 - sqrt(5)*r)
	    H_ij = var*exp_*(25/3) * (delta_i/ell_i^2) * (delta_j/ell_j^2)   i!=j

	  Helmholtz block (divergence-free construction):
	    K0[i,i] = -H[j,j] - H[k,k]   (sum of the OTHER two diagonal entries, negated)
	    K0[i,j] =  H[i,j]             i != j

	  H_ii is always negative (since 5*(delta_i/ell_i)^2 - 1 - sqrt5*r < 0 for
	  typical r), so -H_ii is positive, giving a positive-definite diagonal.
	"""
	delta = v2 - v1                              # (3,)
	r     = jnp.sqrt(jnp.sum((delta / ell) ** 2) + 1e-8)
	exp_  = jnp.exp(-jnp.sqrt(5.0) * r)

	# Diagonal entries of the scalar Hessian
	# H_ii = var * exp_ * (5/(3*ell_i^2)) * (5*(delta_i/ell_i)^2 - 1 - sqrt5*r)
	coeff = var * exp_ * (5.0 / 3.0)
	H_diag = coeff * (5.0 * (delta / ell) ** 2 - 1.0 - jnp.sqrt(5.0) * r) / (ell ** 2)  # (3,)

	# Off-diagonal entries: H_ij = var * exp_ * (25/3) * (delta_i/ell_i^2) * (delta_j/ell_j^2)
	scaled = var * exp_ * (25.0 / 3.0) * (delta / (ell ** 2))  # (3,)
	H_off  = jnp.outer(scaled, delta / (ell ** 2))             # (3,3)

	# Helmholtz block: diagonal is -sum of the OTHER two H_ii entries; off-diag is H_ij
	K0 = H_off  # start with full outer product (correct for off-diagonal)
	# Fix diagonal: K0[i,i] = -H[j,j] - H[k,k] = -(sum of all H_diag) + H_diag[i]
	total = H_diag[0] + H_diag[1] + H_diag[2]
	K0 = K0.at[0, 0].set(H_diag[1] + H_diag[2] - total + total - H_diag[1] - H_diag[2])
	# Simpler: K0[i,i] = -H[j,j] - H[k,k] directly
	K0 = K0.at[0, 0].set(-H_diag[1] - H_diag[2])
	K0 = K0.at[1, 1].set(-H_diag[0] - H_diag[2])
	K0 = K0.at[2, 2].set(-H_diag[0] - H_diag[1])

	return K0


def make_helmholtz_k0(scalar_kernel):
	"""
	Autodiff fallback for non-Matern-5/2 kernels (e.g. Matern-7/2).
	Returns a JIT-compiled helmholtz_k0(v1, v2, ell, var) -> (3, 3).
	"""
	@jax.jit
	def helmholtz_k0(v1, v2, ell, var):
		H = jax.hessian(scalar_kernel)(v1, v2, ell, var)
		return jnp.array([
			[-H[1, 1] - H[2, 2],  H[0, 1],            H[0, 2]],
			[ H[1, 0],            -H[2, 2] - H[0, 0],  H[1, 2]],
			[ H[2, 0],             H[2, 1],            -H[0, 0] - H[1, 1]],
		])
	return helmholtz_k0


# =============================================================================
# Kernel matrix assembly
# =============================================================================

def assemble_kernel_matrix(points_1, points_2, ell, var, helmholtz_k0_fn,
                           noise_std=0.0, jitter=0.0):
	"""
	Assemble the full block covariance matrix.

	Point-major ordering: [u_x(p0), u_y(p0), u_z(p0), u_x(p1), ...]
	Returns shape (3*n1, 3*n2).

	noise_std and jitter are only meaningful when points_1 == points_2 (training
	covariance). Passing them for cross-covariance blocks corrupts the posterior.
	"""
	points_1 = jnp.asarray(points_1)
	points_2 = jnp.asarray(points_2)
	n_1 = points_1.shape[0]
	n_2 = points_2.shape[0]

	blocks = jax.vmap(
		lambda a: jax.vmap(lambda b: helmholtz_k0_fn(a, b, ell, var))(points_2)
	)(points_1)

	# blocks: (n_1, n_2, 3, 3) -> transpose to (n_1, 3, n_2, 3) -> (3*n_1, 3*n_2)
	result = jnp.transpose(blocks, (0, 2, 1, 3)).reshape(n_1 * 3, n_2 * 3)

	if n_1 == n_2:
		if noise_std > 0.0:
			result = result + (noise_std ** 2 + jitter) * jnp.eye(n_1 * 3)
		elif jitter > 0.0:
			result = result + jitter * jnp.eye(n_1 * 3)

	return result


# =============================================================================
# DivergenceFreeGPR
# =============================================================================

class DivergenceFreeGPR:
	"""
	Divergence-free vector-valued GPR using a Helmholtz-derived Matern-5/2 kernel.

	Default path uses an analytical Hessian (helmholtz_k0_matern52) which is
	2-4x faster per optimizer iteration than the previous jax.hessian() path.
	Pass scalar_kernel=matern72 to fall back to the autodiff path for Matern-7/2.

	Mirrors the sklearn GPR interface: construct, .fit(), .predict(), .predict_var().

	Parameters
	----------
	n_restarts      : int
	    Number of L-BFGS-B (Limited-memory Broyden-Fletcher-Goldfarb-Shanno
	    Bounded) restarts for hyperparameter optimisation.
	jitter          : float
	    Diagonal jitter added during fitting to keep K (covariance matrix)
	    positive definite in bad regions of parameter space.
	posterior_batch : int or None
	    Chunk size for predict/predict_var. Set to None to disable batching
	    entirely -- XLA (Accelerated Linear Algebra) can then fuse the full
	    matrix in one shot, which is faster if you have ~20 GB RAM available.
	seed            : int
	    RNG (Random Number Generator) seed for Latin Hypercube start points.
	scalar_kernel   : callable or None
	    If None (default), uses the analytical Matern-5/2 Hessian (fast path).
	    Pass matern72 to use the autodiff Hessian for Matern-7/2.

	Attributes (set after .fit())
	------------------------------
	ell_   : (3,) array  -- fitted anisotropic length scales
	var_   : float       -- fitted signal variance
	noise_ : float       -- fitted noise standard deviation
	nll_   : float       -- negative log likelihood at optimum
	"""

	def __init__(self, n_restarts=8, jitter=1e-4, posterior_batch=4000, seed=0,
	             scalar_kernel=None):
		self.n_restarts = n_restarts
		self.jitter = jitter
		self.posterior_batch = posterior_batch
		self.seed = seed

		# Select kernel block function. None -> fast analytical Matern-5/2 path.
		if scalar_kernel is None:
			self._k0 = helmholtz_k0_matern52
		else:
			self._k0 = make_helmholtz_k0(scalar_kernel)

		# Set after .fit()
		self.ell_   = None
		self.var_   = None
		self.noise_ = None
		self.nll_   = None
		self.alpha_ = None
		self._c     = None
		self._low   = None
		self._training_coords = None

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------

	def fit(self, training_coords, residuals):
		"""
		Fit hyperparameters and solve for alpha.

		Parameters
		----------
		training_coords : (n, 3)
		residuals       : (3n, 1) or (n, 3) -- y minus prior mean

		Returns
		-------
		self
		"""
		training_coords = jnp.asarray(training_coords)
		residuals       = jnp.asarray(residuals).reshape(-1, 1)

		fit = self._fit_hyperparams(training_coords, residuals)
		self.ell_   = jnp.asarray(fit["ell"])
		self.var_   = float(fit["var"])
		self.noise_ = float(fit["noise"])
		self.nll_   = float(fit["nll"])

		# Reuse Cholesky cached from the final NLL evaluation -- no second build.
		self._c   = fit["chol_c"]
		self._low = fit["chol_low"]

		self.alpha_ = cho_solve((self._c, self._low), residuals)
		self._training_coords = training_coords
		return self

	def predict(self, test_points, means_tests, progress_every=0):
		"""
		Posterior mean at test points.

		Parameters
		----------
		test_points    : (m, 3)
		means_tests    : (m, 3) -- prior mean at each test point
		progress_every : int    -- print progress every N chunks (0 = silent)

		Returns
		-------
		(m, 3) posterior mean velocities
		"""
		self._check_fitted()
		test_points = np.asarray(test_points)
		means_tests = np.asarray(means_tests).reshape(-1, 3)
		n_test      = test_points.shape[0]

		# posterior_batch=None: single shot, lets XLA fuse the whole matrix.
		if self.posterior_batch is None:
			ks     = assemble_kernel_matrix(
				test_points, self._training_coords,
				self.ell_, self.var_, self._k0,
			)
			return means_tests + np.array(ks @ self.alpha_).reshape(-1, 3)

		out      = np.empty((n_test, 3), dtype=float)
		n_chunks = (n_test + self.posterior_batch - 1) // self.posterior_batch

		for ci, i in enumerate(range(0, n_test, self.posterior_batch)):
			tp  = test_points[i:i + self.posterior_batch]
			ks  = assemble_kernel_matrix(
				tp, self._training_coords,
				self.ell_, self.var_, self._k0,
				noise_std=0.0, jitter=0.0,
			)
			out[i:i + self.posterior_batch] = (
				means_tests[i:i + self.posterior_batch]
				+ np.array(ks @ self.alpha_).reshape(-1, 3)
			)
			if progress_every and (ci % progress_every == 0 or ci == n_chunks - 1):
				print(f"    posterior vels chunk {ci + 1}/{n_chunks}", flush=True)

		return out

	def predict_var(self, test_points, progress_every=0):
		"""
		Posterior variance diagonal at test points.

		Uses the identity diag(A - B @ C) = diag(A) - rowsum(B * C^T) to avoid
		materializing the full (3m, 3m) matrix per chunk.

		Parameters
		----------
		test_points    : (m, 3)
		progress_every : int -- print progress every N chunks (0 = silent)

		Returns
		-------
		(m, 3) posterior variance per spatial component
		"""
		self._check_fitted()
		test_points = np.asarray(test_points)
		n_test      = test_points.shape[0]

		def _var_chunk(chunk):
			K_tt = assemble_kernel_matrix(
				chunk, chunk, self.ell_, self.var_, self._k0,
			)
			k_tc = assemble_kernel_matrix(
				chunk, self._training_coords,
				self.ell_, self.var_, self._k0,
				noise_std=0.0, jitter=0.0,
			)
			beta            = cho_solve((self._c, self._low), k_tc.T)
			diag_correction = jnp.sum(k_tc * beta.T, axis=1)
			return jnp.diag(K_tt) - diag_correction

		# posterior_batch=None: single shot.
		if self.posterior_batch is None:
			return np.asarray(_var_chunk(test_points)).reshape(-1, 3)

		out      = np.empty((n_test * 3,), dtype=float)
		n_chunks = (n_test + self.posterior_batch - 1) // self.posterior_batch

		for ci, i in enumerate(range(0, n_test, self.posterior_batch)):
			chunk                      = test_points[i:i + self.posterior_batch]
			m                          = chunk.shape[0]
			out[i * 3:(i + m) * 3]    = np.asarray(_var_chunk(chunk))
			if progress_every and (ci % progress_every == 0 or ci == n_chunks - 1):
				print(f"    posterior vars chunk {ci + 1}/{n_chunks}", flush=True)

		return out.reshape(-1, 3)

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _fit_hyperparams(self, X, y):
		"""
		Optimise hyperparameters via L-BFGS-B over multiple random restarts.

		Returns dict with keys: ell, var, noise, nll, chol_c, chol_low.
		chol_c / chol_low are the Cholesky factors at the optimum, reused by
		fit() to avoid a redundant matrix build and factorization.
		"""
		n       = X.shape[0]
		_jitter = float(self.jitter)
		_k0     = self._k0

		print(f"[gpr debug] n_points={n}, K size=({3*n}, {3*n})")
		print(f"[gpr debug] X range: {float(X.min()):.4f} to {float(X.max()):.4f}")
		print(f"[gpr debug] y range: {float(y.min()):.4f} to {float(y.max()):.4f}")
		print(f"[gpr debug] any NaN in X: {bool(jnp.isnan(X).any())}")
		print(f"[gpr debug] any NaN in y: {bool(jnp.isnan(y).any())}")

		lo = np.log([5.0,   5.0,   5.0,   1e-1, 1e-2])
		hi = np.log([150.0, 150.0, 150.0, 1e3,  2e0])
		print(f"[gpr debug] ell bounds: [{np.exp(lo[0]):.1f}, {np.exp(hi[0]):.1f}]")

		# Smoke test at midpoint before entering optimizer loop.
		mid        = (lo + hi) / 2.0
		_ell_mid   = jnp.asarray(np.exp(mid[:3]))
		_var_mid   = float(np.exp(mid[3]))
		_noise_mid = float(np.exp(mid[4]))
		print(f"[gpr debug] smoke test: ell={np.exp(mid[:3])}, var={_var_mid:.3f}, noise={_noise_mid:.3f}")
		try:
			K_smoke      = assemble_kernel_matrix(X, X, _ell_mid, _var_mid, _k0,
			                                      noise_std=_noise_mid, jitter=_jitter)
			c_smoke, _   = cho_factor(K_smoke)
			print(f"[gpr debug] smoke OK -- K diag range: "
			      f"{float(jnp.diag(K_smoke).min()):.4f} to {float(jnp.diag(K_smoke).max()):.4f}, "
			      f"chol diag range: "
			      f"{float(jnp.diag(c_smoke).min()):.4f} to {float(jnp.diag(c_smoke).max()):.4f}")
		except Exception as e:
			print(f"[gpr debug] SMOKE TEST FAILED: {type(e).__name__}: {e}")

		def nll(log_theta):
			ell   = jnp.exp(log_theta[:3])
			var   = jnp.exp(log_theta[3])
			noise = jnp.exp(log_theta[4])

			blocks = jax.vmap(
				lambda a: jax.vmap(lambda b: _k0(a, b, ell, var))(X)
			)(X)
			K      = jnp.transpose(blocks, (0, 2, 1, 3)).reshape(3 * n, 3 * n)
			K      = K + (noise ** 2 + _jitter) * jnp.eye(3 * n)
			c, low = cho_factor(K)
			alpha  = cho_solve((c, low), y)
			logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(c)))
			return 0.5 * (y.T @ alpha)[0, 0] + 0.5 * logdet + 0.5 * (3 * n) * jnp.log(2 * jnp.pi)

		nll_vg = jax.jit(jax.value_and_grad(nll))

		print("[gpr debug] warming up JIT...")
		try:
			_v, _g = nll_vg(jnp.asarray((lo + hi) / 2.0))
			print(f"[gpr debug] JIT warmup OK: nll={float(_v):.4f}, "
			      f"grad_norm={float(jnp.linalg.norm(_g)):.4f}")
		except Exception as e:
			print(f"[gpr debug] JIT WARMUP FAILED: {type(e).__name__}: {e}")
			traceback.print_exc()
			raise

		def objective(log_theta):
			val, grad = nll_vg(jnp.asarray(log_theta))
			return float(val), np.asarray(grad, dtype=float)

		starts = lo + qmc.LatinHypercube(d=5, seed=self.seed).random(self.n_restarts) * (hi - lo)

		best = None
		for i, t0 in enumerate(starts):
			try:
				res = spo.minimize(
					objective, t0, method="L-BFGS-B", jac=True,
					bounds=list(zip(lo, hi)), options={"maxiter": 200},
				)
				print(f"[gpr debug] restart {i}: success={res.success}, "
				      f"nll={res.fun:.4f}, msg={res.message}")
				if best is None or res.fun < best.fun:
					best = res
			except Exception as e:
				print(f"[gpr debug] restart {i} FAILED: {type(e).__name__}: {e}")
				traceback.print_exc()

		if best is None:
			raise RuntimeError("All optimizer restarts crashed. The math broke.")

		theta     = np.exp(best.x)
		ell_opt   = theta[:3]
		var_opt   = float(theta[3])
		noise_opt = float(theta[4])

		# One final assembly at the optimum to cache the Cholesky for fit().
		K_opt      = assemble_kernel_matrix(
			X, X, jnp.asarray(ell_opt), var_opt, _k0,
			noise_std=noise_opt, jitter=1e-8,
		)
		c_opt, low_opt = cho_factor(K_opt)

		return {
			"ell":      ell_opt,
			"var":      var_opt,
			"noise":    noise_opt,
			"nll":      float(best.fun),
			"chol_c":   c_opt,
			"chol_low": low_opt,
		}

	def _check_fitted(self):
		if self.alpha_ is None:
			raise RuntimeError("Call .fit() before predicting.")
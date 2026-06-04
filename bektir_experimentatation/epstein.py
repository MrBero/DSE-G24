import numpy as np
import pyvista
from INTERP import interpolation
from momentum import momentum_closed, momentum_open


PKL_PATH = r"INTERP\csv_with_everything.pkl"
MIDPOINT = np.array([
    -6524.591 / 1000,
    1.213 * 10**5 / 10**3,
    52.6 / 2
])
N_POINTS = 4
SHARPNESS = 2

def sample_cylinder_uniform(
	R: float,
	H: float,
	n: int,
	top: bool = False,
	bottom: bool = False,
) -> np.ndarray:
	"""Sample staggered uniform points on the lateral surface of a cylinder,
	with optional uniform disc sampling on the top and/or bottom caps.
	"""
	# --- Lateral surface ---
	n_cols = max(1, round(np.sqrt(n * 2 * np.pi * R / H)))
	n_rows = max(1, round(n / n_cols))

	idx = np.arange(n_rows * n_cols)
	row_idx = idx // n_cols
	col_idx = idx % n_cols

	theta_shift = (row_idx % 2) * (np.pi / n_cols)
	theta = (2 * np.pi * col_idx / n_cols) + theta_shift
	z = H * row_idx / max(n_rows - 1, 1)

	points = [np.c_[R * np.cos(theta), R * np.sin(theta), z]]

	# --- Cap helper ---
	def sample_disc(z_val: float) -> np.ndarray:
		"""Sunflower/Fibonacci spiral for uniform disc coverage."""
		golden = np.pi * (3.0 - np.sqrt(5.0))  # ~137.5 degrees
		k = np.arange(n)
		r = R * np.sqrt((k + 0.5) / n)
		a = golden * k
		return np.c_[r * np.cos(a), r * np.sin(a), np.full(n, z_val)]

	if bottom:
		points.append(sample_disc(0.0))
	if top:
		points.append(sample_disc(H))

	return np.vstack(points)


region_points = np.array([MIDPOINT[0],MIDPOINT[1],10]) + sample_cylinder_uniform(20,70, n = 80, top=True, bottom=True)
print('x_min, ', np.min(region_points[:,0]))
print('y_min ', np.min(region_points[:,1]))
print('z_min ', np.min(region_points[:,2]))


cloud = pyvista.PolyData(region_points)

print('build interp')
sample = interpolation.build_cfd_sampler(PKL_PATH, n_points=N_POINTS, sharpness=SHARPNESS)

print(sample([[-6524.591 / 1000,1.213 * 10**5 / 10**3,0]]))

print('interp points')
v,p = sample(region_points)

# has_bad = not np.isfinite(v[v != None]).all()
# has_bad2 = not np.isfinite(p[p != None]).all()
print(np.isnan(v).any())
print(np.isnan(p).any())

print('interp done')

print('compute force')
F, mesh = momentum_closed.surface_force(
	points=region_points,
	center=MIDPOINT,
	velocity=v,
	pressure=p,
	rho=1.225,
)



print(F)




arrows = mesh.glyph(orient="Normals", scale=False, factor=5)

plotter = pyvista.Plotter()
plotter.add_mesh(mesh)
plotter.add_mesh(arrows, color="red")
plotter.show()

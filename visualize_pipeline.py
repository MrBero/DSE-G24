"""
visualize_pipeline.py
---------------------
Pretty PyVista render of the wind-force pipeline result:
  • Momentum cylinder surface  – semi-transparent, coloured by pressure
  • Building STL               – fully opaque, muted grey
  • Drone sampling points      – visible spheres coloured by |V|

Call visualize(...) after the main pipeline has run, or run this file
standalone to test with synthetic data (no CFD inputs required).

Dependencies:
	pip install pyvista[all] trimesh numpy
"""

from __future__ import annotations
import numpy as np
import pyvista as pv
import trimesh

# ---------------------------------------------------------------------------
# Palette – dark-background scientific aesthetic
# ---------------------------------------------------------------------------
BG_COLOR         = "#0d0f14"          # near-black
BUILDING_COLOR   = "#9fd3ff"          # light blue
BUILDING_EDGE    = "#ffffff"
GROUND_COLOR     = "#ffffffe0"          # green
AXES_COLOR       = "#ffffff"
SCALAR_BAR_COLOR = "#e0e4ec"
PRESSURE_CMAP    = "RdBu_r"
VELOCITY_CMAP    = "plasma"

POINT_RADIUS_FRAC = 0.004
OPACITY_SURFACE   = 0.28
OPACITY_BUILDING  = 1.0

# ---------------------------------------------------------------------------
# Core visualiser
# ---------------------------------------------------------------------------

def visualize(
	mom_mesh,               # pyvista PolyData/UnstructuredGrid with "pressure" & "velocity"
	stl_mesh,               # trimesh.Trimesh (already scaled to metres)
	drone_pts: np.ndarray,  # (N,3) sampling coordinates
	drone_vels: np.ndarray | None = None,  # (N,3) optional – colours the spheres
	window_size: tuple[int, int] = (1600, 1000),
	screenshot: str | None = None,         # path to save PNG, None = interactive
	camera_position: str | tuple = "iso",
	show: bool = True,
) -> pv.Plotter:
	"""
	Render the pipeline scene.

	Parameters
	----------
	mom_mesh     : PyVista mesh that carries 'pressure' point data and,
				   optionally, 'velocity' (N×3) point data.
	stl_mesh     : trimesh.Trimesh of the building.
	drone_pts    : (N,3) array of sampling coordinates.
	drone_vels   : (N,3) velocity at drone points – used only for colouring.
				   If None, spheres are coloured by a uniform accent.
	window_size  : plotter window pixel dimensions.
	screenshot   : if given, saves a PNG instead of opening the GUI.
	camera_position : pyvista camera preset or explicit ((x,y,z),(fx,fy,fz),(ux,uy,uz)).
	show         : call plotter.show(); set False to embed in a larger scene.
	"""

	# ---- convert trimesh → pyvista PolyData ----------------------------
	verts  = np.asarray(stl_mesh.vertices, dtype=float)
	faces  = np.asarray(stl_mesh.faces, dtype=np.int32)
	pv_faces = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int32), faces])
	building_pv = pv.PolyData(verts, pv_faces.ravel())
	building_pv = building_pv.compute_normals(
	cell_normals=False,
	point_normals=True,
	split_vertices=True,
	feature_angle=30,
)

	# ---- momentum surface: ensure pressure is a point scalar -----------
	mom = mom_mesh.copy()
	if "pressure" not in mom.point_data:
		# fall back to cell data → interpolate
		if "pressure" in mom.cell_data:
			mom = mom.cell_data_to_point_data()

	p_vals = mom.point_data.get("pressure", None)
	if p_vals is None:
		raise KeyError("mom_mesh has no 'pressure' point or cell array.")
	pmin, pmax = float(p_vals.min()), float(p_vals.max())

	# ---- drone spheres -------------------------------------------------
	speeds = (np.linalg.norm(drone_vels, axis=1)
			  if drone_vels is not None and drone_vels.ndim == 2
			  else None)

	scene_diag = float(np.linalg.norm(
		np.asarray(mom.bounds[1::2]) - np.asarray(mom.bounds[::2])
	))
	sphere_r = POINT_RADIUS_FRAC * scene_diag

	# build a single merged glyph cloud for efficiency
	cloud = pv.PolyData(drone_pts.astype(float))
	if speeds is not None:
		cloud["speed"] = speeds.astype(float)
	sphere_proto = pv.Sphere(radius=sphere_r, theta_resolution=12, phi_resolution=8)
	drone_glyphs = cloud.glyph(geom=sphere_proto, scale=False, orient=False)

	# ---- plotter -------------------------------------------------------
	pl = pv.Plotter(window_size=window_size, lighting="three lights")
	pl.set_background(BG_COLOR)

	# momentum surface – semi-transparent, scalar-coloured
	mom_actor = pl.add_mesh(
		mom,
		scalars="pressure",
		cmap=PRESSURE_CMAP,
		clim=(pmin, pmax),
		opacity=OPACITY_SURFACE,
		show_edges=False,
		smooth_shading=True,
		lighting=True,
		scalar_bar_args=dict(
			title="Pressure  [Pa]",
			title_font_size=14,
			label_font_size=11,
			color=SCALAR_BAR_COLOR,
			vertical=True,
			position_x=0.88,
			position_y=0.20,
			width=0.03,
			height=0.55,
			fmt="%.0f",
		),
	)

	# building – opaque light blue
	pl.add_mesh(
		building_pv,
		color=BUILDING_COLOR,
		opacity=OPACITY_BUILDING,
		smooth_shading=True,
		show_edges=False,
		lighting=True,
		ambient=0.25,
		diffuse=0.85,
		specular=0.25,
		specular_power=30,
	)
	# ------------------------------------------------------------------
	# Ground plane
	# ------------------------------------------------------------------
	xmin, xmax, ymin, ymax, _, _ = mom.bounds

	# Make the ground much larger than the visible scene
	scene_size = max(xmax - xmin, ymax - ymin)
	ground_size = scene_size * 1.0     # increase this if you still see edges

	ground = pv.Plane(
		center=(
			(xmin + xmax) / 2,
			(ymin + ymax) / 2,
			-0.01,                     # slightly below the building
		),
		direction=(0, 0, 1),
		i_size=ground_size,
		j_size=ground_size,
	)

	pl.add_mesh(
		ground,
		color=GROUND_COLOR,
		ambient=0.45,
		diffuse=0.85,
		specular=0.02,
	)
	# drone glyphs
	if speeds is not None:
		pl.add_mesh(
			drone_glyphs,
			scalars="speed",
			cmap=VELOCITY_CMAP,
			show_edges=False,
			smooth_shading=True,
			lighting=True,
			scalar_bar_args=dict(
				title="|V|  [m/s]",
				title_font_size=14,
				label_font_size=11,
				color=SCALAR_BAR_COLOR,
				vertical=True,
				position_x=0.04,
				position_y=0.20,
				width=0.03,
				height=0.55,
				fmt="%.1f",
			),
		)
	else:
		pl.add_mesh(
			drone_glyphs,
			color="#f5c842",        # warm gold accent when no velocity data
			show_edges=False,
			smooth_shading=True,
			lighting=True,
		)

	# axes
	pl.add_axes(
		color=AXES_COLOR,
		xlabel="X", ylabel="Y", zlabel="Z",
		line_width=3,
	)

	# optional free-stream arrow annotation
	_add_freestream_arrow(pl, mom, scene_diag)

	# camera
	pl.camera_position = camera_position
	pl.reset_camera()

	if screenshot:
		pl.show(screenshot=screenshot, auto_close=True)
	elif show:
		pl.show()

	return pl


# ---------------------------------------------------------------------------
# Small helper: draw a free-stream direction arrow outside the scene
# ---------------------------------------------------------------------------

def _add_freestream_arrow(pl: pv.Plotter, mom_mesh, scene_diag: float):
	"""Adds a labelled arrow indicating the free-stream direction."""
	bounds = mom_mesh.bounds          # (xmin,xmax, ymin,ymax, zmin,zmax)
	cx = (bounds[0] + bounds[1]) * 0.5
	cy = (bounds[2] + bounds[3]) * 0.5
	cz = (bounds[4] + bounds[5]) * 0.5
	arrow_len = scene_diag * 0.18
	# place arrow to the -Y side (upstream)
	start = np.array([cx - arrow_len * 0.5,
					  bounds[2] - scene_diag * 0.12,
					  cz])
	direction = np.array([0.0, 1.0, 0.0])   # +Y = downstream

	arrow = pv.Arrow(start=start, direction=direction,
					 scale=arrow_len, tip_length=0.25, tip_radius=0.06,
					 shaft_radius=0.025)
	pl.add_mesh(arrow, color="#55aaff", lighting=False)
	pl.add_point_labels(
		[start + direction * arrow_len * 1.15],
		["V∞"],
		font_size=14,
		text_color="#55aaff",
		show_points=False,
		always_visible=True,
	)


# ---------------------------------------------------------------------------
# Standalone test – runs with synthetic data so you can check the look
# without needing any CFD files
# ---------------------------------------------------------------------------

def _synthetic_demo():
	import trimesh as tm

	rng = np.random.default_rng(0)

	# fake cylinder momentum mesh
	cyl = pv.Cylinder(center=(0, 0, 15), direction=(0, 0, 1),
					  radius=25, height=30, resolution=80)
	cyl = cyl.triangulate()
	n   = cyl.n_points
	pts = np.asarray(cyl.points)
	# synthetic pressure: high on windward, low on leeward
	cyl["pressure"] = 0.5 * 1.225 * 13.6**2 * (
		-np.sin(np.arctan2(pts[:, 0], pts[:, 1]))
	)
	cyl["velocity"] = rng.uniform(10, 18, (n, 3))

	# fake building: a box
	box = pv.Box(bounds=(-6, 6, -6, 6, 0, 24))
	box = box.triangulate()
	building = tm.creation.box(extents=[12, 12, 24])
	building.apply_translation([0, 0, 12])

	# fake drone cloud: points on the cylinder surface ± small jitter
	theta = rng.uniform(0, 2 * np.pi, 300)
	z_d   = rng.uniform(1, 29, 300)
	r_d   = 25 + rng.normal(0, 0.5, 300)
	d_pts = np.column_stack([r_d * np.cos(theta),
							 r_d * np.sin(theta),
							 z_d])
	d_vel = rng.uniform(8, 20, (300, 3))

	visualize(
		mom_mesh   = cyl,
		stl_mesh   = building,
		drone_pts  = d_pts,
		drone_vels = d_vel,
		screenshot = None,   # change to "output.png" to save
	)


if __name__ == "__main__":
	_synthetic_demo()
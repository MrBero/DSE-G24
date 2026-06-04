
import sys
import numpy as np
import trimesh
import matplotlib.pyplot as plt


# 4-point degree-3 triangle quadrature (barycentric) ------------------------- #
_BARY = np.array([[1/3, 1/3, 1/3],
                  [0.6, 0.2, 0.2],
                  [0.2, 0.6, 0.2],
                  [0.2, 0.2, 0.6]])
_WTS = np.array([-27/48, 25/48, 25/48, 25/48])


# =========================================================================== #
#  Automatic mesh conditioning                                                 #
# =========================================================================== #
def condition_mesh(mesh, target_edge_frac=0.06, max_panels=3000, verbose=True):
    """
    Make an arbitrary input mesh suitable for panel solving.

    1. merge duplicate vertices / drop degenerate + duplicate faces
    2. fix normals to point consistently outward
    3. adaptively subdivide faces whose longest edge exceeds
       target_edge_frac * body_diagonal, until none remain or the panel cap
       is reached.
    """
    m = mesh.copy()
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    try:
        m.fix_normals()
    except Exception:
        pass

    diag = float(np.linalg.norm(m.extents))
    target = target_edge_frac * diag

    def longest_edges(mm):
        tri = mm.triangles
        e = np.stack([
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        ], axis=1)
        return e.max(axis=1)

    for _ in range(12):
        le = longest_edges(m)
        too_big = le > 1.5 * target          # hysteresis: only clearly-coarse faces
        # stop if nothing coarse, or if subdividing would blow the cap (each
        # flagged face -> 4 faces)
        projected = m.faces.shape[0] + 3 * int(np.sum(too_big))
        if not np.any(too_big) or projected > max_panels:
            break
        v, f = trimesh.remesh.subdivide(
            m.vertices, m.faces, face_index=np.where(too_big)[0])
        m = trimesh.Trimesh(vertices=v, faces=f, process=True)
        try:
            m.fix_normals()
        except Exception:
            pass

    if verbose:
        print(f"  conditioned mesh: {m.faces.shape[0]} panels "
              f"(target edge {target:.3g}, body diag {diag:.3g}, "
              f"watertight={m.is_watertight})")
    return m


# =========================================================================== #
#  Solver                                                                      #
# =========================================================================== #
class VortexSheetSolver:
    def __init__(self, mesh, V_inf, auto_condition=True,
                 target_edge_frac=0.06, max_panels=3000, verbose=True):
        self.raw_mesh = mesh
        self.mesh = condition_mesh(self.raw_mesh, target_edge_frac, max_panels, verbose) \
            if auto_condition else self.raw_mesh
        self.V_inf = np.asarray(V_inf, dtype=float)

        self.tris    = np.asarray(self.mesh.triangles)
        self.centers = np.asarray(self.mesh.triangles_center)
        self.normals = np.asarray(self.mesh.face_normals)
        self.areas   = np.asarray(self.mesh.area_faces)
        self.N = self.tris.shape[0]

        A, B, C = self.tris[:, 0], self.tris[:, 1], self.tris[:, 2]
        self.qpts = (A[:, None, :] * _BARY[None, :, 0, None] +
                     B[:, None, :] * _BARY[None, :, 1, None] +
                     C[:, None, :] * _BARY[None, :, 2, None])
        self.qwts = _WTS

        # geometry-derived post-processing scales
        self.diag = float(np.linalg.norm(self.mesh.extents))
        self.mask_dist = 0.015 * self.diag

        self.sigma = self._solve()

    def _panel_velocity(self, field_pts):
        diff = field_pts[:, None, None, :] - self.qpts[None, :, :, :]
        r = np.linalg.norm(diff, axis=-1)
        r = np.maximum(r, 1e-12)
        kern = diff / (4.0 * np.pi * r[..., None] ** 3)
        return np.einsum('pnqk,q->pnk', kern, self.qwts) * self.areas[None, :, None]

    def _solve(self, chunk=512):
        # Assemble A row-block by row-block so we never hold the full
        # (N, N, Q, 3) tensor in memory at once.
        N = self.N
        A = np.empty((N, N))
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            vel = self._panel_velocity(self.centers[s:e])     # (b, N, 3)
            A[s:e] = np.einsum('ink,ik->in', vel, self.normals[s:e])
        np.fill_diagonal(A, 0.5)                 # exact flat-panel self term
        rhs = -self.normals @ self.V_inf
        return np.linalg.solve(A, rhs)

    def velocity(self, pts, blank_interior=True, blank_near=True, chunk=None):
        pts = np.asarray(pts, dtype=float)
        P = pts.shape[0]
        if chunk is None:
            # keep the (chunk, N, Q, 3) working array near ~50M floats
            chunk = max(64, int(12_000_000 / max(self.N, 1)))
        out = np.tile(self.V_inf, (P, 1)).astype(float)
        for s in range(0, P, chunk):
            e = min(s + chunk, P)
            vel = self._panel_velocity(pts[s:e])
            out[s:e] += np.einsum('pnk,n->pk', vel, self.sigma)
        if blank_interior:
            try:
                out[self.mesh.contains(pts)] = np.nan
            except Exception:
                pass
        if blank_near:
            try:
                from trimesh.proximity import closest_point
                _, dist, _ = closest_point(self.mesh, pts)
                out[dist < self.mask_dist] = np.nan
            except Exception:
                pass
        return out

    def bc_residual(self):
        vel = self._panel_velocity(self.centers)
        idx = np.arange(self.N)
        vel[idx, idx, :] = 0.0
        induced = np.einsum('ink,n->ik', vel, self.sigma)
        vn = (np.einsum('ik,ik->i', induced, self.normals)
              + 0.5 * self.sigma + self.normals @ self.V_inf)
        return float(np.abs(vn).max())

    def net_source(self):
        return float(np.sum(self.sigma * self.areas))

    def generate_flow_field(self, x, y, z, zero_inside=True):
        """
        Evaluate total velocity on a meshgrid.
        x, y, z : arrays of identical shape (any meshgrid indexing)
        returns : (N, 3) ordered like np.stack([x, y, z], -1).reshape(-1, 3)
        """
        self.grid_points = np.stack([x, y, z], axis=-1)
        grid_points = self.grid_points.reshape(-1, 3)
        # Do NOT NaN-blank here: the original set interior velocities to 0.0,
        # and downstream griddata/GP code expects finite numbers everywhere.
        vel = self.velocity(
            grid_points, blank_interior=False, blank_near=False)
        if zero_inside:
            try:
                inside = self.mesh.contains(grid_points)
                vel[inside] = 0.0
            except Exception:
                pass
        return vel

    # convenience pass-throughs
    def bc_residual(self):
        return self.bc_residual()

    def net_source(self):
        return self.net_source()

# =========================================================================== #
#  Automatic slice plotting (all params derived from geometry)                 #
# =========================================================================== #
def plot_slice(solver, axis='z', frac=0.5, n=160, pad_frac=0.8,
               ax=None, title=None, clip_pct=99.0):
    lo, hi = solver.mesh.bounds
    span = hi - lo
    pad = pad_frac * span
    lo2, hi2 = lo - pad, hi + pad
    ai = {'x': 0, 'y': 1, 'z': 2}[axis]
    a0, a1 = [i for i in range(3) if i != ai]

    g0 = np.linspace(lo2[a0], hi2[a0], n)
    g1 = np.linspace(lo2[a1], hi2[a1], n)
    G0, G1 = np.meshgrid(g0, g1, indexing='xy')
    coord = lo[ai] + frac * span[ai]

    pts = np.zeros((G0.size, 3))
    pts[:, a0] = G0.ravel(); pts[:, a1] = G1.ravel(); pts[:, ai] = coord

    vel = solver.velocity(pts)
    speed = np.linalg.norm(vel, axis=1).reshape(G0.shape)
    Va = vel[:, a0].reshape(G0.shape)
    Vb = vel[:, a1].reshape(G0.shape)

    finite = speed[np.isfinite(speed)]
    vmax = np.percentile(finite, clip_pct) if finite.size else 1.0
    vmax = max(vmax, np.linalg.norm(solver.V_inf) * 1.05)

    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 6))

    pc = ax.contourf(G0, G1, np.clip(speed, 0, vmax), levels=40,
                     cmap='viridis', vmin=0, vmax=vmax)
    plt.colorbar(pc, ax=ax, label='|velocity|')

    mask = ~np.isfinite(speed)
    ax.streamplot(g0, g1, np.where(mask, 0.0, Va), np.where(mask, 0.0, Vb),
                  color='white', density=1.5, linewidth=0.7, arrowsize=0.8)

    origin = [0.0, 0.0, 0.0]; origin[ai] = coord
    try:
        sec = solver.mesh.section(plane_origin=origin, plane_normal=np.eye(3)[ai])
        if sec is not None:
            p2, _ = sec.to_2D()
            for ent in p2.entities:
                v = p2.vertices[ent.points]
                ax.fill(v[:, 0], v[:, 1], color='0.25', zorder=5)
                ax.plot(v[:, 0], v[:, 1], 'k-', lw=1.2, zorder=6)
    except Exception:
        pass

    labels = ['x', 'y', 'z']
    ax.set_xlabel(labels[a0]); ax.set_ylabel(labels[a1])
    ax.set_aspect('equal')
    ax.set_xlim(lo2[a0], hi2[a0]); ax.set_ylim(lo2[a1], hi2[a1])
    ax.set_title(title or f'{axis}={coord:.3g} slice')
    return ax


# =========================================================================== #
#  Backward-compatible wrapper for the original main.py interface              #
# =========================================================================== #
class PotentialFlowSolver:

    def __init__(self, V_inf, stl_mesh, n_vortices_per_tri=1,
                 auto_condition=True, verbose=True):
        # NOTE: original signature is (V_inf, mesh, ...). Keep that order.
        self._solver = VortexSheetSolver(
            stl_mesh, V_inf, auto_condition=auto_condition, verbose=verbose)
        self.V_inf = self._solver.V_inf
        self.mesh = self._solver.mesh          # exposed for main.py's scatter plot
        self.n = n_vortices_per_tri            # kept only for compatibility

    def generate_flow_field(self, x, y, z, zero_inside=True):
        """
        Evaluate total velocity on a meshgrid.
        x, y, z : arrays of identical shape (any meshgrid indexing)
        returns : (N, 3) ordered like np.stack([x, y, z], -1).reshape(-1, 3)
        """
        grid_points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        # Do NOT NaN-blank here: the original set interior velocities to 0.0,
        # and downstream griddata/GP code expects finite numbers everywhere.
        vel = self._solver.velocity(
            grid_points, blank_interior=False, blank_near=False)
        if zero_inside:
            try:
                inside = self.mesh.contains(grid_points)
                vel[inside] = 0.0
            except Exception:
                pass
        return vel

    # convenience pass-throughs
    def bc_residual(self):
        return self._solver.bc_residual()

    def net_source(self):
        return self._solver.net_source()


def auto_visualize(solver, savepath=None, show=False):
    """
    Choose slices automatically using BOTH the flow direction and geometry.

    Wind direction picks the most informative cuts:
      * a plane PERPENDICULAR to the flow's dominant axis at the body centre
        -> shows how flow wraps the cross-section it actually sees;
      * a plane CONTAINING the flow (normal = a transverse axis) at centre
        -> shows fore/aft stagnation and the over-body deflection;
      * a second perpendicular-to-flow slice offset toward one end.
    All positions are fractions of the body extent, so they scale to any size.
    """
    name = {0: 'x', 1: 'y', 2: 'z'}
    V = solver.V_inf
    flow_axis = int(np.argmax(np.abs(V))) if np.linalg.norm(V) > 0 else 0
    ext = solver.mesh.extents

    # For a clear "flow wraps the body" picture we want a slice plane that
    # CONTAINS the flow vector (so streamlines stay in-plane) and is normal to
    # the axis the body is most extruded along. That normal is the largest
    # extent among the non-flow axes -> the cross-section the flow really sees.
    non_flow = [i for i in range(3) if i != flow_axis]
    wrap_normal = non_flow[int(np.argmax(ext[non_flow]))]   # in-plane: flow + other axis
    # A complementary plane, also containing the flow, normal = the remaining axis
    other_normal = [i for i in non_flow if i != wrap_normal][0]

    fig, axs = plt.subplots(1, 3, figsize=(19, 5.8))
    plot_slice(solver, axis=name[wrap_normal], frac=0.5, ax=axs[0],
               title=f'flow-plane (perp {name[wrap_normal]}, mid): wraps section')
    plot_slice(solver, axis=name[other_normal], frac=0.5, ax=axs[1],
               title=f'flow-plane (perp {name[other_normal]}, mid): stagnation & wake')
    plot_slice(solver, axis=name[wrap_normal], frac=0.2, ax=axs[2],
               title=f'flow-plane (perp {name[wrap_normal]}, off-centre)')
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=110)
    if show:
        plt.show()
    return fig




def main():
    stl = trimesh.load_mesh('input_stls/triangle.stl')

    print(f"\n=== Arbitrary geometry: {stl} ===")
    solver = VortexSheetSolver(stl, [10.0, 0.0, 0.0])
    rel = solver.net_source() / (np.linalg.norm(solver.V_inf) * solver.mesh.area)
    print(f"  BC residual {solver.bc_residual():.2e}   "
          f"net source {solver.net_source():.2e}   (rel {rel:.2%})")
    auto_visualize(solver, savepath='fig_geometry.png')
    print("fig_geometry.png saved")
    plt.show()


if __name__ == "__main__":
    main()
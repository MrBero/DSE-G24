import matplotlib.pyplot as plt
import numpy as np
import trimesh


class PotentialFlowSolver():
    def __init__(self, V_inf, stl_mesh):
        self.mesh = stl_mesh
        self.mesh.fix_normals()

        self.mesh_centers = self.mesh.triangles_center
        self.mesh_normals = self.mesh.face_normals
        self.mesh_triangles = self.mesh.triangles

        self.V_inf = np.asarray(V_inf, dtype=float)

        # Precompute triangle edges (for vortex rings)
        self.edges = self._triangle_edges()

        # Solve for circulation strengths
        self.gammas = self.solve_gamma()

    # --------------------------------------------------
    # Geometry
    # --------------------------------------------------
    def _triangle_edges(self):
        tris = self.mesh_triangles

        edges = np.stack([
            np.stack([tris[:, 0], tris[:, 1]], axis=1),
            np.stack([tris[:, 1], tris[:, 2]], axis=1),
            np.stack([tris[:, 2], tris[:, 0]], axis=1),
        ], axis=1)

        return edges  # (N_tri, 3, 2, 3)

    # --------------------------------------------------
    # Biot–Savart kernel (finite vortex segment)
    # --------------------------------------------------
    @staticmethod
    def vortex_segment_velocity(P, A, B, gamma):
        r1 = P - A
        r2 = P - B

        r1_norm = np.linalg.norm(r1, axis=1)
        r2_norm = np.linalg.norm(r2, axis=1)

        r1_norm = np.maximum(r1_norm, 1e-8)
        r2_norm = np.maximum(r2_norm, 1e-8)

        cross = np.cross(r1, r2)
        dot = np.sum(r1 * r2, axis=1)

        denom = (r1_norm * r2_norm * (r1_norm * r2_norm + dot))
        denom = np.maximum(denom, 1e-12)

        coeff = (gamma / (4 * np.pi)) * ((r1_norm + r2_norm) / denom)

        vel = cross * coeff[:, None]

        return vel

    def vortex_ring_velocity(self, P, tri_edges, gamma):
        vel = np.zeros((P.shape[0], 3))

        for edge in tri_edges:
            A, B = edge
            vel += self.vortex_segment_velocity(P, A, B, gamma)

        return vel

    # --------------------------------------------------
    # Solve linear system
    # --------------------------------------------------
    def solve_gamma(self):
        N = self.mesh_centers.shape[0]
        A = np.zeros((N, N))

        for i in range(N):
            ni = self.mesh_normals[i]
            Pi = self.mesh_centers[i:i+1]

            for j in range(N):
                if i == j:
                    A[i, j] = 0.5
                    continue

                vel = self.vortex_ring_velocity(Pi, self.edges[j], gamma=1.0)
                A[i, j] = np.dot(vel[0], ni)

        RHS = -np.dot(self.mesh_normals, self.V_inf)

        print("Condition number:", np.linalg.cond(A))

        gammas = np.linalg.solve(A, RHS)
        return gammas

    # --------------------------------------------------
    # Flow field
    # --------------------------------------------------
    def generate_flow_field(self, x, y, z):
        self.grid_points = np.stack([x, y, z], axis=-1)
        grid_points = self.grid_points.reshape(-1, 3)

        vel = np.tile(self.V_inf, (grid_points.shape[0], 1))

        for j in range(len(self.edges)):
            vel += self.vortex_ring_velocity(grid_points, self.edges[j], self.gammas[j])

        # Mask interior
        try:
            is_inside = self.mesh.contains(grid_points)
            vel[is_inside] = 0
        except:
            pass

        return vel

    def plot_slice(self, vel, slice):
        vel_grid = vel.reshape(*x.shape, -1)
        Xs, Ys = self.grid_points[:,:,slice,0], self.grid_points[:,:,slice,1]
        us, vs, ws = vel_grid[:,:,slice,0], vel_grid[:,:,slice,1], vel_grid[:,:,slice,2]
        mag = np.sqrt(us**2 + vs**2 + ws**2)

        fig, ax2 = plt.subplots(figsize=(7, 6))
        pc = ax2.contourf(Xs, Ys, mag, levels=30, cmap='viridis')
        fig.colorbar(pc, ax=ax2, label='|velocity|')
        ax2.scatter(*self.edges[:,:2].T, c='black')
        ax2.quiver(Xs, Ys, us, vs, color='white')
        ax2.set_aspect('equal')
        # ax2.set_title(f'xy slice at z={np.linspace(bounds[2,0],bounds[2,1],res)[k]:.2f}')

    def plot_3D(self, vel):
        grid = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(*self.edges.T, c='black')
        ax.quiver(*grid.T, *vel.T, length = 1)
        ax.set_aspect('equal')
        # plt.show()

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    res = 30

    # Use METERS (important!)
    V_inf = np.array([10.0, 0.0, 0.0])

    x_dim, y_dim, z_dim = (4.0, 5.0, 7.0)

    x, y, z = np.meshgrid(
        np.linspace(-x_dim/2, x_dim/2, res),
        np.linspace(-y_dim/2, y_dim/2, res),
        np.linspace(0, z_dim, res),
        indexing='ij'
    )

    stl_mesh = trimesh.load_mesh('inputs/triangle.stl')
    stl_mesh.apply_scale(1/1000)

    solver = PotentialFlowSolver(V_inf, stl_mesh)
    flowvel = solver.generate_flow_field(x, y, z)
    solver.plot_slice(flowvel, slice = x.shape[0] //2)
    plt.show()
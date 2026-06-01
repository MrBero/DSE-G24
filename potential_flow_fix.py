import matplotlib.pyplot as plt
import numpy as np
import trimesh


class PotentialFlowSolver():
    def __init__(self, V_inf, stl_mesh, n_vortices_per_tri=4):
        self.mesh = stl_mesh
        self.mesh_centers = self.mesh.triangles_center 
        self.mesh_normals = self.mesh.face_normals
        self.mesh_triangles = self.mesh.triangles  # (N_tri, 3, 3)
        self.V_inf = V_inf
        self.n = n_vortices_per_tri

        # Generate vortex sheet points and their associated normals
        self.vortex_points, self.vortex_normals, self.tri_indices = \
            self._generate_vortex_sheet_points()

        self.gammas = self.gammas_VPM(
            self.mesh_centers,
            self.mesh_normals,
            self.vortex_points,
            self.vortex_normals,
            V_inf=self.V_inf
        )

    def _generate_vortex_sheet_points(self):
        triangles = self.mesh_triangles      # (N_tri, 3, 3)
        normals   = self.mesh_normals        # (N_tri, 3)
        N_tri = triangles.shape[0]
        n = self.n

        # --- stratified barycentric coordinates ---
        # Build a grid of (u, v) such that u + v <= 1
        # For n points, use a triangular lattice with ~sqrt(2n) steps
        k = int(np.ceil((-1 + np.sqrt(1 + 8 * n)) / 2))  # largest k s.t. k*(k+1)/2 <= n
        coords = []
        for i in range(k + 1):
            for j in range(k + 1 - i):
                coords.append((i / k, j / k))
                if len(coords) == n:
                    break
            if len(coords) == n:
                break

        # Pad or trim to exactly n points
        while len(coords) < n:
            # Fill remaining slots with centroid
            coords.append((1/3, 1/3))
        coords = np.array(coords[:n])  # (n, 2)

        u = coords[:, 0]  # (n,)
        v = coords[:, 1]  # (n,)
        w = 1.0 - u - v   # (n,)  barycentric third coord

        # Clamp any numerical negatives
        w = np.clip(w, 0, None)
        # Renormalise
        total = u + v + w
        u, v, w = u / total, v / total, w / total

        # Convert barycentric -> 3D:  p = u*A + v*B + w*C
        # triangles: (N_tri, 3, 3)  ->  A=(N_tri,3), B=(N_tri,3), C=(N_tri,3)
        A = triangles[:, 0, :]  # (N_tri, 3)
        B = triangles[:, 1, :]
        C = triangles[:, 2, :]

        # Broadcast: (N_tri, 1, 3) * (1, n, 1)  -> (N_tri, n, 3)
        points_3d = (
            A[:, None, :] * u[None, :, None] +
            B[:, None, :] * v[None, :, None] +
            C[:, None, :] * w[None, :, None]
        )  # (N_tri, n, 3)

        vortex_points  = points_3d.reshape(-1, 3)              # (N_tri*n, 3)
        vortex_normals = np.repeat(normals, n, axis=0)         # (N_tri*n, 3)
        tri_indices    = np.repeat(np.arange(N_tri), n)        # (N_tri*n,)

        return vortex_points, vortex_normals, tri_indices
    
    @staticmethod
    def _solid_angle(centroid, tri_vertices):
        a, b, c = tri_vertices[0], tri_vertices[1], tri_vertices[2]
        ra = a - centroid
        rb = b - centroid
        rc = c - centroid

        ra_n = np.linalg.norm(ra)
        rb_n = np.linalg.norm(rb)
        rc_n = np.linalg.norm(rc)

        # Numerator: scalar triple product
        numerator = np.dot(ra, np.cross(rb, rc))
        # Denominator
        denominator = (ra_n * rb_n * rc_n
                       + np.dot(ra, rb) * rc_n
                       + np.dot(rb, rc) * ra_n
                       + np.dot(rc, ra) * rb_n)

        # arctan2 to get the signed solid angle; multiply by 2
        omega = 2.0 * np.arctan2(numerator, denominator)
        return abs(omega)

    def gammas_VPM(self, centroids, normals, vortex_points, vortex_normals, V_inf):
        N_tri = centroids.shape[0]
        n     = self.n
        A     = np.zeros((N_tri, N_tri))

        for i in range(N_tri):
            ni = normals[i]
            for j in range(N_tri):
                if i == j:
                    omega = self._solid_angle(centroids[i],
                                              self.mesh_triangles[i])
                    A[i, j] = omega / (4.0 * np.pi)
                    continue

                # Indices of the n vortex points belonging to triangle j
                mask = (self.tri_indices == j)
                vp   = vortex_points[mask]        # (n, 3)
                influence = 0.0
                for k in range(n):
                    r_vect = centroids[i] - vp[k]
                    r      = np.linalg.norm(r_vect)
                    if r < 1e-12:
                        continue
                    influence += np.dot(r_vect / (4 * np.pi * r**3), ni)
                A[i, j] = influence

        RHS    = -np.dot(normals, V_inf)
        gammas = np.linalg.solve(A, RHS)
        return gammas

    def uniform_flow_velocity(self, V_inf):
        """
        Return the uniform free-stream velocity vector (same everywhere).
        This is separate from the scalar potential — the velocity is V_inf,
        not a scalar function of position.
        """
        return np.asarray(V_inf, dtype=float)

    def source_velocity(self, grid_points, x0, gamma):
        r_vec = grid_points - x0[None, :]          # (M, 3)
        r2    = np.sum(r_vec**2, axis=1)            # (M,)
        # Regularise to avoid division by zero at the source location
        r2    = np.maximum(r2, 1e-12)
        r3    = r2 * np.sqrt(r2)                    # |r|^3,  shape (M,)
        return gamma * r_vec / (4.0 * np.pi * r3[:, None])  # (M, 3)

    def generate_flow_field(self, x, y, z):
        self.grid_points = np.stack([x, y, z], axis=-1)
        grid_points = self.grid_points.reshape(-1, 3)

        is_inside = self.mesh.contains(grid_points)                 # (M,)

        vel = np.tile(self.V_inf.astype(float), (grid_points.shape[0], 1))  # (M, 3)

        for k, point in enumerate(self.vortex_points):
            j       = self.tri_indices[k]
            gamma_k = self.gammas[j] / self.n      # sub-point strength
            vel    += self.source_velocity(grid_points, point, gamma_k)

        # Zero velocity inside the body
        # print(is_inside)
        vel[is_inside, :] = np.zeros(((is_inside).sum(), 3))
        # print(vel)
        return vel

    def plot_slice(self, vel, slice):
        vel_grid = vel.reshape(*self.grid_points.shape)
        Xs, Ys = self.grid_points[:,:,slice,0], self.grid_points[:,:,slice,1]
        us, vs, ws = vel_grid[:,:,slice,0], vel_grid[:,:,slice,1], vel_grid[:,:,slice,2]
        mag = np.sqrt(us**2 + vs**2 + ws**2)

        fig, ax2 = plt.subplots(figsize=(7, 6))
        pc = ax2.contourf(Xs, Ys, mag, levels=30, cmap='viridis')
        fig.colorbar(pc, ax=ax2, label='|velocity|')
        ax2.scatter(*self.vortex_points[:,:2].T, c='black')
        ax2.quiver(Xs, Ys, us, vs, color='white')
        ax2.set_aspect('equal')
        ax2.set_title(f'Potential Flow xy slice at {slice}')

    def plot_3D(self, vel):
        grid = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(*self.vortex_points.T, c='black')
        ax.quiver(*grid.T, *vel.T, length = 1)
        ax.set_aspect('equal')
        # plt.show()


if __name__ == "__main__":
    res = 40
    V_inf = np.array([10.0, 0.0, 0.0])
    # x_dim = 15 * 1000 #must be in mm
    # y_dim = 45 * 1000
    # z_dim = 15 * 1000

    # x_dim, y_dim, z_dim = (15000, 45000, 15000) #must be in mm
    x_dim, y_dim, z_dim = (4, 5, 7)
    x, y, z = np.meshgrid(np.linspace(-x_dim/2, x_dim/2, res),
                          np.linspace(-y_dim/2, y_dim/2, res),
                          np.linspace(0, z_dim, res),
                           indexing='ij') 

    stl_mesh = trimesh.load_mesh('inputs/triangle.stl')
    stl_mesh.apply_scale(1/1000) #to m
    potentialFlowSolve = PotentialFlowSolver(V_inf, stl_mesh,
                                             n_vortices_per_tri=5)
    flowvel = potentialFlowSolve.generate_flow_field(x, y, z)
    potentialFlowSolve.plot_slice(flowvel, slice = x.shape[0] //2)
    plt.show()
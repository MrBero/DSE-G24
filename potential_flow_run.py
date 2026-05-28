import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
# from numba import njit
import trimesh

# @njit
class PotentialFlowSolver():
    def __init__(self, V_inf, stl_filepath):
        self.mesh = trimesh.load_mesh(stl_filepath)
        self.mesh_centers = self.mesh.triangles_center
        self.mesh_normals = self.mesh.face_normals
        self.V_inf = V_inf
        self.gammas = self.gammas_VPM(self.mesh_centers, self.mesh_normals, V_inf=self.V_inf)

    def gammas_VPM(self, centroids, normals, V_inf): #vortex panel method
        n = centroids.shape[0]
        A = np.zeros((n,n))

        for i in range(n): #we target i and collect influences from all other mesh panels on it into a matrix row
            for j in range(n):
                if i==j:
                    A[i,j] = 0.5
                    continue
                r_vect = centroids[i] - centroids[j]
                r = np.linalg.norm(r_vect)

                A[i,j] = np.dot(r_vect / (4 * np.pi * r**3), normals[i])
        
        RHS = -np.dot(normals, V_inf)

        gammas = np.linalg.solve(A, RHS)
        return gammas

    def uniform_flow(self, x,y,z,U,V,W):
        return U*x + V*y + W*z

    def point_vortex(self, x,y,z,x0,y0,z0,gamma):
        return -gamma/(4*np.pi * np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2))

    def generate_flow_field(self, x, y, z):
        #vel stream collection
        self.vel_stream = self.uniform_flow(x,y,z, *self.V_inf)
        for i, point in enumerate(self.mesh_centers):
            self.vel_stream += self.point_vortex(x, y, z, *point, gamma=self.gammas[i])

        #take gradient to get velocities
        u, v, w = jnp.gradient(self.vel_stream,
                            x[:,0,0],
                            y[0,:,0],
                            z[0,0,:])

        u_f = u.ravel()
        v_f = v.ravel()
        w_f = w.ravel()

        # full_grid = np.stack([x,y,z], axis=-1).reshape(-1,3)
        # # print(full_grid.shape)
        # outside_bool = self.mesh.contains(full_grid)
        # # print(sum(outside_bool))
        # inside_bool = outside_bool.__invert__()
        # grid = full_grid[inside_bool,:]
        # x_q = grid[:,0]; y_q = grid[:,1]; z_q = grid[:,2]
        # u_q = u_f[inside_bool]; v_q = v_f[inside_bool]; w_q = w_f[inside_bool] 

        # ax = plt.figure().add_subplot(projection='3d')
        # ax.scatter3D(self.mesh_centers[:,0], self.mesh_centers[:,1], self.mesh_centers[:,2], c='black', alpha=0.5)
        # ax.quiver(x_q,y_q,z_q,
        #         u_q,v_q,w_q,
        #         length=0.1, normalize=True)

        # plt.show()
        return np.vstack([u_f,v_f,w_f]).T

if __name__ == "__main__":
    res = 20
    V_inf = np.array([10,0,0])
    x, y, z = np.meshgrid(np.linspace(-2, 2, res),
                        np.linspace(-2, 2, res),
                        np.linspace(-2, 2, res),
                        indexing='ij')

    potentialFlowSolve = PotentialFlowSolver(V_inf,'inputs/sphere.stl')
    flowvel = potentialFlowSolve.generate_flow_field(x,y,z)

    print(flowvel.shape)

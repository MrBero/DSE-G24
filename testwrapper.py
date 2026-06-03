import numpy as np
import trimesh
import matplotlib.pyplot as plt
from flowpanelwrapper import FLOWPanelSolver

# 1. Load your geometry (Replace with your actual file, or use a test shape)
print("Loading mesh...")
mesh = trimesh.creation.icosphere(radius=1.0) # OR: trimesh.load("my_wing.stl")

# 2. Initialize the solver (Set wind speed to 10 m/s in the X direction)
# Make sure 'julia_script' exactly matches your Julia file name!
print("Starting solver...")
solver = FLOWPanelSolver(
    mesh=mesh, 
    V_inf=[10.0, 0.0, 0.0], 
    julia_script="FP.jl" 
)

# 3. Create a grid of points to probe the flow
# Here, we create a 2D slice down the middle (Y=0) of the object
print("Setting up observation grid...")
x = np.linspace(-3, 3, 20)  # 20 points from X=-3 to 3
z = np.linspace(-3, 3, 20)  # 20 points from Z=-3 to 3
X, Z = np.meshgrid(x, z)
Y = np.zeros_like(X)        # Keep Y completely flat at 0

# 4. Predict the flow field!
print("Solving flow field (calling Julia)...")
velocities = solver.generate_flow_field(X.flatten(), Y.flatten(), Z.flatten())

# 'velocities' is an (N, 3) array. Let's extract the X and Z components for plotting
U = velocities[:, 0].reshape(X.shape)
W = velocities[:, 2].reshape(Z.shape)

# 5. Visualize the result
print("Plotting...")
plt.figure(figsize=(8, 6))
plt.title("Flow Field Velocity Vectors")
plt.quiver(X, Z, U, W, color='blue')

# Draw a rough circle to represent where our sphere is
circle = plt.Circle((0, 0), 1.0, color='gray', alpha=0.5)
plt.gca().add_patch(circle)

plt.xlabel("X")
plt.ylabel("Z")
plt.axis("equal")
plt.show()
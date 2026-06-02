import os
import numpy as np
import trimesh
from flowpanelwrapper import FLOWPanelSolver

def run_tests():
    # 1. Setup mock geometry and conditions
    print("Generating a test mesh (icosphere)...")
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    
    # Freestream velocity: 10 m/s in the X direction
    v_inf = [10.0, 0.0, 0.0]
    
    print(f"Initializing FLOWPanelSolver with V_inf = {v_inf}...")
    solver = FLOWPanelSolver(
        mesh=mesh,
        V_inf=v_inf,
        julia_script="FP.jl",  # Make sure this file exists in your working directory
        verbose=True
    )

    # 2. Test `velocity()` method
    print("\n--- Testing velocity() ---")
    # Point 1: Far outside the sphere (flow should be close to V_inf)
    # Point 2: Right at the origin (inside the sphere, should return NaN)
    test_points = np.array([
        [0.0, 5.0, 0.0], 
        [0.0, 0.0, 0.0]   
    ])
    
    vel = solver.velocity(test_points, blank_interior=True)
    for pt, v in zip(test_points, vel):
        print(f"Point {pt} -> Velocity {v}")

    # 3. Test `generate_flow_field()` method
    print("\n--- Testing generate_flow_field() ---")
    # Create a small 3x3x3 grid around the sphere
    grid_coords = np.linspace(-2, 2, 3)
    x, y, z = np.meshgrid(grid_coords, grid_coords, grid_coords)
    
    grid_vel = solver.generate_flow_field(
        x.flatten(), 
        y.flatten(), 
        z.flatten(), 
        zero_inside=True
    )
    
    print(f"Successfully generated flow field for {len(x.flatten())} points.")
    print(f"Sample of first 3 velocities:\n{grid_vel[:3]}")

if __name__ == "__main__":
    # Sanity check: verify the Julia script exists before running
    if not os.path.exists("FP.jl"):
        print("⚠️ Warning: 'FP.jl' not found in the current directory.")
        print("The solver will likely crash when it calls subprocess.run().")
        print("Make sure FP.jl is present or update the path in the test script!\n")
        
    try:
        run_tests()
        print("\n✅ Wrapper executed successfully.")
    except Exception as e:
        print(f"\n❌ Wrapper test failed: {e}")
# FP.jl
# Usage: julia FP.jl <stl_path> <Vx> <Vy> <Vz> <pts_npy> <out_npy>

import FLOWPanel as pnl
import Meshes
import FileIO, MeshIO
import NPZ

stl_path, Vx, Vy, Vz, pts_file, out_file = ARGS
V_inf = [parse(Float64, Vx), parse(Float64, Vy), parse(Float64, Vz)]

# --- load mesh ---
raw_mesh = FileIO.load(stl_path)
points = [Meshes.Point(Float64(p[1]), Float64(p[2]), Float64(p[3])) for p in raw_mesh.position]
connec = [Meshes.connect((Int(f[1].i) + 1, Int(f[2].i) + 1, Int(f[3].i) + 1), Meshes.Triangle) for f in raw_mesh.faces]
msh    = Meshes.SimpleMesh(points, connec)

grid = pnl.gt.GridTriangleSurface(msh)
body = pnl.NonLiftingBody{pnl.ConstantSource}(grid)

println("Panels: $(body.ncells)")

# --- solve ---
Uinfs = repeat(V_inf, 1, body.ncells)
pnl.solve(body, Uinfs)

# --- evaluate velocity at requested points ---
pts = NPZ.npzread(pts_file)           # (N, 3) float64
N   = size(pts, 1)

# BYPASS BUG: Convert the target points to a 3xN Matrix.
# FLOWPanel handles batch-matrices flawlessly and much faster than single points.
targets = Matrix(pts')                  # (3, N) Matrix
U_inds  = zeros(Float64, 3, N)          # (3, N) Matrix

# Calculate induced velocities for ALL points simultaneously
pnl.Uind!(body, targets, U_inds)

# Add freestream velocity (broadcasts automatically to each column)
# and transpose back to (N, 3) for the Python wrapper to read
out = Matrix((U_inds .+ V_inf)')

NPZ.npzwrite(out_file, out)
println("Saved $(N) velocities → $(out_file)")
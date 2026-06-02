# flowpanel_solve.jl
# Usage: julia flowpanel_solve.jl <stl_path> <Vx> <Vy> <Vz> <pts_npy> <out_npy>

import FLOWPanel as pnl
import Meshes
import FileIO, MeshIO
import NPZ

stl_path, Vx, Vy, Vz, pts_file, out_file = ARGS
V_inf = [parse(Float64, Vx), parse(Float64, Vy), parse(Float64, Vz)]

# --- load mesh ---
# Bypassing GeoIO to avoid the GDAL fallback error.
# We read the STL natively and construct a Meshes.jl object.
raw_mesh = FileIO.load(stl_path)

# Extract vertices (points) and faces (connectivity)
points = [Meshes.Point(Float64(p[1]), Float64(p[2]), Float64(p[3])) for p in raw_mesh.position]
connec = [Meshes.connect((Int(f[1]), Int(f[2]), Int(f[3])), Meshes.Triangle) for f in raw_mesh.faces]
msh    = Meshes.SimpleMesh(points, connec)

# FLOWPanel's GeometricTools natively accepts the built Meshes.jl object
grid = pnl.gt.GridTriangleSurface(msh)
body = pnl.NonLiftingBody{pnl.ConstantSource}(grid)

println("Panels: $(body.ncells)")

# --- solve ---
Uinfs = repeat(V_inf, 1, body.ncells)
pnl.solve(body, Uinfs)

# --- evaluate velocity at requested points ---
pts = NPZ.npzread(pts_file)           # (N, 3) float64
N   = size(pts, 1)
out = zeros(Float64, N, 3)

for i in 1:N
    X   = pts[i, :]
    vel = pnl.Uind(body, X) .+ V_inf  # induced + freestream
    out[i, :] .= vel
end

NPZ.npzwrite(out_file, out)
println("Saved $(N) velocities → $(out_file)")
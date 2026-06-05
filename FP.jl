# FP.jl
#
# Two modes:
#   One-shot: julia FP.jl <stl> <Vx> <Vy> <Vz> <pts_npy> <out_npy>
#   Server:   julia FP.jl <stl> <Vx> <Vy> <Vz> --server
#
# Server protocol (little-endian, point-major x0,y0,z0,x1,...):
#   read  Int64 N
#   read  N*3 Float64
#   write Int64 N
#   write N*3 Float64
#   N <= 0 means shut down.

import FLOWPanel as pnl
import Meshes
import FileIO, MeshIO
import NPZ

function build_body(stl_path, V_inf)
    raw_mesh = FileIO.load(stl_path)
    points = [Meshes.Point(Float64(p[1]), Float64(p[2]), Float64(p[3]))
              for p in raw_mesh.position]
    connec = [Meshes.connect((Int(f[1].i) + 1, Int(f[2].i) + 1, Int(f[3].i) + 1),
                             Meshes.Triangle) for f in raw_mesh.faces]
    msh  = Meshes.SimpleMesh(points, connec)
    grid = pnl.gt.GridTriangleSurface(msh)
    body = pnl.NonLiftingBody{pnl.ConstantSource}(grid)
    Uinfs = repeat(V_inf, 1, body.ncells)
    pnl.solve(body, Uinfs)
    return body
end

function eval_velocity(body, pts::AbstractMatrix{Float64}, V_inf)
    N = size(pts, 1)
    targets = Matrix(pts')              # (3, N)
    U_inds  = zeros(Float64, 3, N)
    pnl.Uind!(body, targets, U_inds)
    return Matrix((U_inds .+ V_inf)')   # (N, 3)
end

function run_oneshot(stl_path, V_inf, pts_file, out_file)
    body = build_body(stl_path, V_inf)
    println(stderr, "Panels: $(body.ncells)")
    pts = NPZ.npzread(pts_file)
    out = eval_velocity(body, pts, V_inf)
    NPZ.npzwrite(out_file, out)
    println(stderr, "Saved $(size(pts, 1)) velocities -> $(out_file)")
end

function run_server(stl_path, V_inf)
    body = build_body(stl_path, V_inf)
    println(stderr, "Panels: $(body.ncells)")
    println(stderr, "FP.jl server ready")
    flush(stderr)

    # Use the raw binary handles explicitly. On Windows the default stdout can
    # apply CRLF translation, which would corrupt the Float64 byte stream; the
    # raw stdin/stdout below avoid any text-mode mangling on every platform.
    inp  = stdin
    outp = stdout

    while true
        nbytes = read(inp, 8)
        length(nbytes) < 8 && break
        N = only(reinterpret(Int64, nbytes))
        N <= 0 && break

        nb = N * 3 * 8
        raw = read(inp, nb)
        length(raw) < nb && break

        flat = collect(reinterpret(Float64, raw))      # 3N
        pts  = permutedims(reshape(flat, 3, N))         # (N, 3)
        out  = eval_velocity(body, pts, V_inf)          # (N, 3)
        out_flat = vec(permutedims(out))                # x0,y0,z0,...

        write(outp, reinterpret(UInt8, [Int64(N)]))
        write(outp, reinterpret(UInt8, collect(out_flat)))
        flush(outp)
    end

    println(stderr, "FP.jl server shutting down")
    flush(stderr)
end

# --- entry point ---
stl_path = ARGS[1]
V_inf = [parse(Float64, ARGS[2]), parse(Float64, ARGS[3]), parse(Float64, ARGS[4])]

if length(ARGS) >= 5 && ARGS[5] == "--server"
    run_server(stl_path, V_inf)
else
    pts_file, out_file = ARGS[5], ARGS[6]
    run_oneshot(stl_path, V_inf, pts_file, out_file)
end

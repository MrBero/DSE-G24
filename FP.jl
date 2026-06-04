# FP.jl
#
# Two modes:
#
# 1. ONE-SHOT (backwards compatible-ish; freestream now via .npy, not argv):
#       julia FP.jl <stl_path> <vinf_npy> <pts_npy> <out_npy>
#    Boots Julia, solves panels, evaluates one batch of points, writes out_npy,
#    exits. Slow if called repeatedly (full boot + solve every time).
#
# 2. SERVER (fast, default for the Python wrapper):
#       julia FP.jl --server <stl_path>
#    Boots Julia ONCE, solves panels ONCE, then loops reading point batches from
#    stdin and writing velocities to stdout using a length-delimited binary
#    protocol. The Python wrapper keeps this process alive for its whole life.
#
# IMPORTANT: NO numeric values are ever passed as command-line text. The
# freestream V_inf travels as binary Float64 (over stdin in server mode, via a
# .npy file in one-shot mode). This is deliberate: command-line floats are
# parsed under the host C locale, and on a comma-decimal locale (Dutch, German,
# French, ...) "0.5" can round-trip to "0,5" and back to 0.0 in Julia's parse,
# silently changing the physics. Binary floats are locale-immune.
#
# Protocol (server mode), all little-endian, raw bytes on STDOUT/STDIN:
#   - On successful startup the server prints exactly one line to STDOUT:
#         "READY <npanels>\n"
#     and from then on STDOUT carries ONLY binary protocol bytes.
#   - All human-readable logging goes to STDERR, never STDOUT.
#   - Request  (Python -> Julia, on STDIN):
#         magic   : 4 bytes  = "FPQ1"
#         n        : Int64    (number of points)
#         payload  : n*3 Float64, row-major (point-major): x0,y0,z0,x1,y1,z1,...
#         To shut down cleanly Python sends magic "FPXX" and closes stdin.
#   - Response (Julia -> Python, on STDOUT):
#         status   : 1 byte (0x00 = ok, 0x01 = error)
#         if ok:
#             n        : Int64
#             payload  : n*3 Float64, point-major: vx0,vy0,vz0,...
#         if error:
#             msglen   : Int64
#             msg      : msglen bytes UTF-8
#
# Endianness is forced to little-endian on the wire so Windows/Linux/Mac agree.

import FLOWPanel as pnl
import Meshes
import FileIO, MeshIO
import NPZ
import Pkg
import LinearAlgebra

# Print the exact versions of the packages that determine the physics, plus
# thread/BLAS info, to STDERR. If Windows and Linux disagree on results, diff
# these two banners first — a different FLOWPanel/Meshes version is the most
# likely cause of a *large* discrepancy.
function log_environment()
    try
        deps = Pkg.dependencies()
        function ver(name)
            for (_, info) in deps
                info.name == name && return string(info.version)
            end
            return "?"
        end
        println(stderr, "ENV julia=$(VERSION) " *
            "FLOWPanel=$(ver("FLOWPanel")) Meshes=$(ver("Meshes")) " *
            "MeshIO=$(ver("MeshIO")) FileIO=$(ver("FileIO")) NPZ=$(ver("NPZ"))")
    catch e
        println(stderr, "ENV (version query failed: $(sprint(showerror, e)))")
    end
    println(stderr, "ENV threads=$(Threads.nthreads()) " *
        "blas_threads=$(try string(LinearAlgebra.BLAS.get_num_threads()) catch; "?" end)")
    flush(stderr)
end

# ---------------------------------------------------------------------------
# Shared: build + solve the panel body once.
# ---------------------------------------------------------------------------

function build_body(stl_path::AbstractString, V_inf::Vector{Float64})
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

# Evaluate induced velocity for an (N,3) array of points; returns (N,3).
function eval_velocity(body, V_inf::Vector{Float64}, pts::AbstractMatrix{Float64})
    N = size(pts, 1)
    targets = Matrix(pts')               # (3, N)
    U_inds  = zeros(Float64, 3, N)       # (3, N)
    pnl.Uind!(body, targets, U_inds)
    return Matrix((U_inds .+ V_inf)')    # (N, 3)
end

# ---------------------------------------------------------------------------
# Little-endian binary IO helpers (force byte order so OSes agree).
# ---------------------------------------------------------------------------

@inline function read_exact(io::IO, n::Int)
    buf = Vector{UInt8}(undef, n)
    nread = readbytes!(io, buf, n)
    nread == n || error("short read: wanted $n bytes, got $nread (stream closed?)")
    return buf
end

read_i64_le(io::IO) = ltoh(reinterpret(Int64, read_exact(io, 8))[1])

function read_f64_vec_le(io::IO, count::Int)
    raw = read_exact(io, 8 * count)
    v = reinterpret(Float64, raw)          # host order
    @inbounds for i in eachindex(v)
        v[i] = ltoh(v[i])                  # ensure little-endian -> host
    end
    return collect(v)
end

function write_i64_le(io::IO, x::Integer)
    write(io, reinterpret(UInt8, [htol(Int64(x))]))
end

function write_f64_vec_le(io::IO, v::AbstractVector{Float64})
    out = similar(v)
    @inbounds for i in eachindex(v)
        out[i] = htol(v[i])
    end
    write(io, reinterpret(UInt8, out))
end

# ---------------------------------------------------------------------------
# Server loop.
# ---------------------------------------------------------------------------

function run_server(stl_path)
    # Hand the raw stdio streams; STDOUT is binary-only from here on.
    sin  = stdin
    sout = stdout

    # First thing on the wire: the freestream as 3 little-endian Float64s.
    # Passing it as binary (not as command-line text) means it can never be
    # corrupted by the host locale (e.g. comma decimal separators on a Dutch
    # or German Windows box turning "0.5" into "0,5"). No string parsing at all.
    V_inf = read_f64_vec_le(sin, 3)

    log_environment()
    body = build_body(stl_path, V_inf)

    # Signal readiness on STDOUT as one text line, then flush.
    write(sout, "READY $(body.ncells)\n")
    flush(sout)

    while true
        # Read 4-byte magic. EOF -> clean exit.
        magic = Vector{UInt8}(undef, 4)
        got = readbytes!(sin, magic, 4)
        if got == 0
            break                          # stdin closed, parent gone
        elseif got != 4
            break
        end

        if magic == UInt8['F','P','X','X']
            break                          # explicit shutdown
        elseif magic != UInt8['F','P','Q','1']
            # Unknown frame: report error and keep going is unsafe; bail.
            err = "bad request magic: $(String(copy(magic)))"
            write(sout, UInt8(0x01)); write_i64_le(sout, length(err))
            write(sout, Vector{UInt8}(err)); flush(sout)
            break
        end

        try
            n = read_i64_le(sin)
            n >= 0 || error("negative point count $n")
            flat = read_f64_vec_le(sin, Int(n) * 3)
            # point-major flat -> (N,3)
            pts = permutedims(reshape(flat, 3, Int(n)))   # (N,3)
            vel = eval_velocity(body, V_inf, pts)          # (N,3)
            # back to point-major flat
            outflat = vec(permutedims(vel))                # length 3N
            write(sout, UInt8(0x00))
            write_i64_le(sout, n)
            write_f64_vec_le(sout, outflat)
            flush(sout)
        catch e
            msg = sprint(showerror, e)
            try
                write(sout, UInt8(0x01))
                write_i64_le(sout, length(msg))
                write(sout, Vector{UInt8}(msg))
                flush(sout)
            catch
            end
            # An error mid-protocol leaves the stream desynced; safest to exit
            # so the Python side falls back / restarts cleanly.
            break
        end
    end
end

# ---------------------------------------------------------------------------
# One-shot mode (original behavior, file handshake).
# ---------------------------------------------------------------------------

function run_oneshot(stl_path, vinf_file, pts_file, out_file)
    # Freestream comes in as a 3-element .npy, never as command-line text,
    # so it is immune to host-locale number formatting.
    V_inf = Vector{Float64}(vec(NPZ.npzread(vinf_file)))
    length(V_inf) == 3 || error("expected 3-element freestream, got $(length(V_inf))")
    log_environment()
    body = build_body(stl_path, V_inf)
    pts  = NPZ.npzread(pts_file)            # (N,3) float64
    vel  = eval_velocity(body, V_inf, Matrix{Float64}(pts))
    NPZ.npzwrite(out_file, vel)
    println(stderr, "Saved $(size(pts,1)) velocities -> $(out_file)")
end

# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

function main()
    if length(ARGS) >= 1 && ARGS[1] == "--server"
        # Usage: julia FP.jl --server <stl_path>
        # The freestream is read as binary from stdin, not from ARGS.
        stl_path = ARGS[2]
        run_server(stl_path)
    else
        # Usage: julia FP.jl <stl_path> <vinf_npy> <pts_npy> <out_npy>
        stl_path, vinf_file, pts_file, out_file = ARGS
        run_oneshot(stl_path, vinf_file, pts_file, out_file)
    end
end

main()
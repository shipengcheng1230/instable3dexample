using Distributed
addprocs(20)
@everywhere using Quaycle

using GmshTools
using Gmsh_SDK_jll
using HDF5
using LinearAlgebra
using Test

begin
    amsh = "dom04a.msh"
    nx, ny, nz = 80, 3, 3
    rfzn = ones(nz)
    rfzh = accumulate((x, y) -> x * y, fill(1.0, length(rfzn))) |> cumsum |> x -> normalize(x, Inf)
    gen_gmsh_mesh(Val(:BoxHexExtrudeFromSurface), -40e3, -20e3, -10e3, 40.0e3, 40.0e3, 40.0e3, nx, ny, 1.0, 1.0, rfzn, rfzh; filename=amsh)

    ma = read_gmsh_mesh(Val(:SBarbotHex8), joinpath(@__DIR__, "dom04a.msh"))
    ϵcomp = (:xx, :xy, :xz, :yy, :yz, :zz)
    σcomp = (:xx, :xy, :xz, :yy, :yz, :zz)
    vv = stress_greens_func(ma, 3e10, 3e10, ϵcomp, σcomp)
    nume = length(ma.tag)
    @test [all(diag(vv[i][i]) .<= 0) for i in 1: 6] |> all

    vv6 = vcat(
        hcat(vv[1][1], vv[2][1], vv[3][1], vv[4][1], vv[5][1], vv[6][1]),
        hcat(vv[1][2], vv[2][2], vv[3][2], vv[4][2], vv[5][2], vv[6][2]),
        hcat(vv[1][3], vv[2][3], vv[3][3], vv[4][3], vv[5][3], vv[6][3]),
        hcat(vv[1][4], vv[2][4], vv[3][4], vv[4][4], vv[5][4], vv[6][4]),
        hcat(vv[1][5], vv[2][5], vv[3][5], vv[4][5], vv[5][5], vv[6][5]),
        hcat(vv[1][6], vv[2][6], vv[3][6], vv[4][6], vv[5][6], vv[6][6]),
    )

    fname = "GF-$(nx)-$(ny)-$(nz).h5"
    if isfile(fname)
        rm(fname)
    end
    h5write(fname, "vv6", vv6)
end

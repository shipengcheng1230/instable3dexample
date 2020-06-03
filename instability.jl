using HDF5
using Quaycle
using LinearAlgebra
using GmshTools
using Gmsh_SDK_jll
using RecursiveArrayTools
using PyPlot
using Test
using PyCall

inset_axes = pyimport("mpl_toolkits.axes_grid.inset_locator").inset_axes

amsh = "dom04a.msh"
nx, ny, nz = 8, 8, 8
rfzn = ones(nz)
rfzh = accumulate((x, y) -> x * y, fill(1.0, length(rfzn))) |> cumsum |> x -> normalize(x, Inf)
gen_gmsh_mesh(Val(:BoxHexExtrudeFromSurface), -40e3, -20e3, -10e3, 40.0e3, 40.0e3, 40.0e3, nx, ny, 1.0, 1.0, rfzn, rfzh; filename=amsh)

ma = read_gmsh_mesh(Val(:SBarbotHex8), joinpath(@__DIR__, "dom04a.msh"))
ϵcomp = (:xx, :xy, :xz, :yy, :yz, :zz)
σcomp = (:xx, :xy, :xz, :yy, :yz, :zz)
vv = stress_greens_func(ma, 3e10, 3e10, ϵcomp, σcomp)
nume = length(ma.tag)
@test [all(diag(vv[i][i]) .<= 0) for i in 1: 6] |> all

function relax_allcomponents(du, u, p, t, vv, η)
    ϵ, σ = u.x
    dϵ, dσ = du.x
    if maximum(abs.(dϵ)) ≥ 1e-8
        @info t
    end
    for i in 1: size(σ, 1)
        σkk = (σ[i,1] + σ[i,4] + σ[i,6]) / 3
        for j in 1: 6
            if j == 1 || j == 4 || j == 6
                dϵ[i,j] = η * (σ[i,j] - σkk)
            else
                dϵ[i,j] = η * σ[i,j]
            end
        end
    end
    mul!(vec(dσ), vv, vec(dϵ), true, false)
end

function relax_onlyallpartial(du, u, p, t, vv, η)
    ϵ, σ = u.x
    dϵ, dσ = du.x
    dϵ .= η .* σ
    mul!(vec(dσ), vv, vec(dϵ), true, false)
end

relax_singlecomponent = relax_onlyallpartial

vvpartial = vcat(
    hcat(vv[2][2], vv[3][2], vv[5][2]),
    hcat(vv[2][3], vv[3][3], vv[5][3]),
    hcat(vv[2][5], vv[3][5], vv[5][5]),
    )

vv6 = vcat(
    hcat(vv[1][1], vv[2][1], vv[3][1], vv[4][1], vv[5][1], vv[6][1]),
    hcat(vv[1][2], vv[2][2], vv[3][2], vv[4][2], vv[5][2], vv[6][2]),
    hcat(vv[1][3], vv[2][3], vv[3][3], vv[4][3], vv[5][3], vv[6][3]),
    hcat(vv[1][4], vv[2][4], vv[3][4], vv[4][4], vv[5][4], vv[6][4]),
    hcat(vv[1][5], vv[2][5], vv[3][5], vv[4][5], vv[5][5], vv[6][5]),
    hcat(vv[1][6], vv[2][6], vv[3][6], vv[4][6], vv[5][6], vv[6][6]),
)
h5write("GF-$(nx)-$(ny)-$(nz).h5", "vv6", vv6)

begin
    ϵ0 = zeros(nume, 6)
    σ0 = ones(nume, 6) * 1e7 # away from steady state
    u0 = ArrayPartition(ϵ0, σ0)
    f = (du, u, p, t) -> relax_allcomponents(du, u, p, t, vv6, 1e-17)
    prob = ODEProblem(f, u0, (0.0, 15. * 365 * 86400))
    handler(u::ArrayPartition, t, integrator) = (integrator(integrator.t, Val{1}).x[1], u.x[1], u.x[2])
    output_ = joinpath(@__DIR__, "relax-all-$(nx)-$(ny)-$(nz).h5")
    @time wsolve(prob, VCABM5(), output_, 1000, handler, ["dϵ", "ϵ", "σ"], "t"; reltol=1e-8, abstol=1e-8, dt=1e-6, stride=1, maxiters=1e9, force=true)
end

t = h5read(output_, "t")
de = h5read(output_, "dϵ")

maximum(abs.(de))

begin
    # ϵ0 = zeros(nume, 3)
    # σ0 = ones(nume, 3) * 1e7 # away from steady state
    # u0 = ArrayPartition(ϵ0, σ0)
    # f = (du, u, p, t) -> relax_onlyallpartial(du, u, p, t, vvpartial, 1e-17)
    # prob = ODEProblem(f, u0, (0.0, 15. * 365 * 86400))
    # handler(u::ArrayPartition, t, integrator) = (integrator(integrator.t, Val{1}).x[1], u.x[1], u.x[2])
    # output_ = joinpath(@__DIR__, "relax-partial-6.h5")
    # @time wsolve(prob, VCABM5(), output_, 1000, handler, ["dϵ", "ϵ", "σ"], "t"; reltol=1e-8, abstol=1e-8, dt=1e-6, dtmax=10.0*365*86400, stride=1, maxiters=1e9, force=true)
end

begin
    # fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=true)
    # t = h5read("relax-all-6.h5", "t") / 365 / 86400
    # dϵ = h5read("relax-all-6.h5", "dϵ")
    # demax = dropdims(maximum(abs.(dϵ), dims=1), dims=1)

    # ax[1].plot(t, demax[1,:], label=raw"$\dot{ϵ} _{\mathrm{xx}}$", linestyle="-", linewidth=3)
    # ax[1].plot(t, demax[2,:], label=raw"$\dot{ϵ} _{\mathrm{xy}}$", linestyle=":", linewidth=3)
    # ax[1].plot(t, demax[3,:], label=raw"$\dot{ϵ} _{\mathrm{xz}}$", linestyle=":", linewidth=3)
    # ax[1].plot(t, demax[4,:], label=raw"$\dot{ϵ} _{\mathrm{yy}}$", linestyle="-", linewidth=3)
    # ax[1].plot(t, demax[5,:], label=raw"$\dot{ϵ} _{\mathrm{yz}}$", linestyle=":", linewidth=3)
    # ax[1].plot(t, demax[6,:], label=raw"$\dot{ϵ} _{\mathrm{zz}}$", linestyle="-", linewidth=3)
    # ax[1].legend(loc="upper left", ncol=3)
    # ax[1].set_xlim([0, 15])
    # ax[1].set_ylim([0, 1e-8])
    # ax[1].set_ylabel("Strain Rate (1/sec)")

    # axi = inset_axes(ax[1], width="55%", height="55%", loc=4)
    # axi.plot(t, demax[1,:], label=raw"$\dot{ϵ} _{\mathrm{xx}}$", linestyle="-", linewidth=1)
    # axi.plot(t, demax[2,:], label=raw"$\dot{ϵ} _{\mathrm{xy}}$", linestyle=":", linewidth=1)
    # axi.plot(t, demax[3,:], label=raw"$\dot{ϵ} _{\mathrm{xz}}$", linestyle=":", linewidth=1)
    # axi.plot(t, demax[4,:], label=raw"$\dot{ϵ} _{\mathrm{yy}}$", linestyle="-", linewidth=1)
    # axi.plot(t, demax[5,:], label=raw"$\dot{ϵ} _{\mathrm{yz}}$", linestyle=":", linewidth=1)
    # axi.plot(t, demax[6,:], label=raw"$\dot{ϵ} _{\mathrm{zz}}$", linestyle="-", linewidth=1)
    # axi.set_xlim([0, 1.5])
    # axi.set_ylim([0, 1e-10])

    # t = h5read("relax-partial-6.h5", "t") / 365 / 86400
    # dϵ = h5read("relax-partial-6.h5", "dϵ")
    # demax = dropdims(maximum(dϵ, dims=1), dims=1)

    # ax[2].plot(t, demax[1,:], label=raw"$\dot{ϵ} _{\mathrm{xy}}$", linestyle=":", linewidth=3)
    # ax[2].plot(t, demax[2,:], label=raw"$\dot{ϵ} _{\mathrm{xz}}$", linestyle=":", linewidth=3)
    # ax[2].plot(t, demax[3,:], label=raw"$\dot{ϵ} _{\mathrm{yz}}$", linestyle=":", linewidth=3)
    # ax[2].legend(loc="upper left", ncol=3)
    # ax[2].set_xlim([0, 15])
    # ax[2].set_xlabel("Year")
    # ax[2].set_ylabel("Strain Rate (1/sec)")
    # fig.savefig(joinpath(@__DIR__, "strain-rate.pdf"), bbox_inches="tight")
    # plt.close("all")
end

# cache = gmsh_vtk_output_cache("dom04a.msh", 3, -1)
# dϵ = h5read("relax-all-6.h5", "dϵ")
# t = h5read("relax-all-6.h5", "t")
# de = permutedims(dϵ, [2, 1, 3])
# vtk_output("movie/relax-all-instable", t, [de], ["strain rate"], cache)
# maximum(dϵ)
# t[end]/365/86400

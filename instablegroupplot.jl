using HDF5
using Quaycle
using LinearAlgebra
using GmshTools
using Gmsh_SDK_jll
using RecursiveArrayTools
using PyPlot
using Test
using PyCall

group1 = [
    [2,2,2],[3,3,3],[4,4,4],
    [5,5,5],[6,6,6],[7,7,7],
    [8,8,8],[9,9,9],[10,10,10],
]

group2 = [
    [2,1,1],[4,2,2],[6,3,3],
    [8,4,4],[10,5,5],[12,6,6],
    [14,7,7],[16,8,8],
]

group3 = [
    [1,8,8],[2,8,8],[4,8,8],
    [6,8,8],[8,8,8],[10,8,8],
    [12,8,8],[16,8,8],[20,8,8],
    [24,8,8],
]

group4 = [
    [3,3,3],[6,3,3],[10,3,3],
    [20,3,3],[40,3,3],[60,3,3],
    [80,3,3],[100,3,3],
]

function write_vtk_mesh(g)
    for gg in g
        nx, ny, nz = gg
        amsh = "vtk-$(nx)-$(ny)-$(nz).vtk"
        rfzn = ones(nz)
        rfzh = accumulate((x, y) -> x * y, fill(1.0, length(rfzn))) |> cumsum |> x -> normalize(x, Inf)
        gen_gmsh_mesh(Val(:BoxHexExtrudeFromSurface), -40e3, -20e3, -10e3, 40.0e3, 40.0e3, 40.0e3, nx, ny, 1.0, 1.0, rfzn, rfzh; filename=amsh)
    end
end

write_vtk_mesh(group1)
write_vtk_mesh(group2)
write_vtk_mesh(group3)
write_vtk_mesh(group4)

function relax_allcomponents(du, u, p, t, vv, η)
    ϵ, σ = u.x
    dϵ, dσ = du.x
    if maximum(abs.(dϵ)) ≥ 1e-8
        @info "instable begin: $(t)"
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

function calculate_member(gg, ηeff, yearto)
    nx, ny, nz = gg
    vv6 = h5read("GF-$(nx)-$(ny)-$(nz).h5", "vv6")
    nume = size(vv6, 1) ÷ 6
    ϵ0 = zeros(nume, 6)
    σ0 = ones(nume, 6) * 1e7 # away from steady state
    u0 = ArrayPartition(ϵ0, σ0)
    f = (du, u, p, t) -> relax_allcomponents(du, u, p, t, vv6, ηeff)
    prob = ODEProblem(f, u0, (0.0, yearto))
    handler(u::ArrayPartition, t, integrator) = (integrator(integrator.t, Val{1}).x[1], u.x[1], u.x[2])
    output_ = joinpath(@__DIR__, "relax-all-$(nx)-$(ny)-$(nz).h5")
    @time wsolve(prob, VCABM5(), output_, 1000, handler, ["dϵ", "ϵ", "σ"], "t"; reltol=1e-8, abstol=1e-8, dt=1e-6, stride=1, maxiters=1e9, force=true)
end

function calculate_group(g, ηeff=1e-17, yearto=5. * 365 * 86400)
    for gg in g
        @info gg
        calculate_member(gg, ηeff, yearto)
    end
end

calculate_group(group1)
calculate_group(group2)
calculate_group(group3)
calculate_group(group4, 1e-17, 25. * 365 * 86400)

function plot_group(g, fname, comp=1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for gg in g
        nx, ny, nz = gg
        f = "relax-all-$(nx)-$(ny)-$(nz).h5"
        t = h5read(f, "t")
        de = h5read(f, "dϵ")
        dexx = de[:,comp,:] # 1 for xx
        dexxmax = dropdims(maximum(dexx; dims=1), dims=1)
        ax.plot(t/365/86400, dexxmax, label="$(nx)-$(ny)-$(nz)")
    end
    ax.set_ylim([-1e-10, 1e-9])
    ax.set_xlabel("Year")
    ax.set_ylabel("Strain Rate (1/sec)")
    fig.legend(loc="upper center", ncol=3)
    fig.savefig(joinpath(@__DIR__, fname), bbox_inches="tight")
    plt.close("all")
end

plot_group(group1, "max-dexx-g1.pdf")
plot_group(group2, "max-dexx-g2.pdf")
plot_group(group3, "max-dexx-g3.pdf")
plot_group(group4, "max-dexx-g4.pdf")
plot_group(group4, "max-deyy-g4.pdf", 4)
plot_group(group4, "max-dezz-g4.pdf", 6)

de = h5read("relax-partial-6.h5", "dϵ")
de[:,:,end] |> x -> maximum(abs.(x))

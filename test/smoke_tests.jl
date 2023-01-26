@testset "Smoke Tests" begin
    mts = Type{<:ICNF.AbstractICNF}[RNODE, FFJORD, Planar]
    cmts = Type{<:ICNF.AbstractCondICNF}[CondRNODE, CondFFJORD, CondPlanar]
    ats = Type{<:AbstractArray}[Array]
    if has_cuda_gpu()
        push!(ats, CuArray)
    end
    tps = Type{<:AbstractFloat}[Float64, Float32, Float16]
    nvars_ = (1:2)

    if GROUP == "All" || GROUP == "Call"
        include("call_tests.jl")
    end
    if GROUP == "All" || GROUP == "Fit"
        include("fit_tests.jl")
    end
end

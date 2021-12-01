using
    ICNF,
    CUDA,
    ComputationalResources,
    DataFrames,
    Distributions,
    Flux,
    MLJBase,
    Test

CUDA.allowscalar() do
    include("core.jl")

    @testset "Overall" begin
        include("ffjord.jl")
        include("rnode.jl")

        include("cond_ffjord.jl")
        include("cond_rnode.jl")
    end
end

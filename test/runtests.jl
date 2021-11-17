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
    end
end

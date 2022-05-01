using
    ICNF,
    CUDA,
    ComputationalResources,
    DataFrames,
    Distributions,
    Flux,
    ForwardDiff,
    GalacticOptim,
    MLJBase,
    SciMLBase,
    Test,
    Zygote

CUDA.allowscalar() do
    include("core.jl")

    @testset "Overall" begin
        include("smoke_tests.jl")
    end
end

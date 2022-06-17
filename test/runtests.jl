using
    ICNF,
    AbstractDifferentiation,
    CUDA,
    ComputationalResources,
    DataFrames,
    Distributions,
    FiniteDifferences,
    Flux,
    ForwardDiff,
    Optimization,
    ReverseDiff,
    MLJBase,
    SciMLBase,
    Test,
    Tracker,
    Zygote

CUDA.allowscalar() do
    include("core.jl")

    @testset "Overall" begin
        include("smoke_tests.jl")
    end
end

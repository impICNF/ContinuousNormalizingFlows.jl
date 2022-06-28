using
    ICNF,
    AbstractDifferentiation,
    CUDA,
    ComputationalResources,
    DataFrames,
    Distributions,
    FiniteDiff,
    FiniteDifferences,
    Flux,
    ForwardDiff,
    Optimization,
    ReverseDiff,
    MLJBase,
    SciMLBase,
    Test,
    Tracker,
    Yota,
    Zygote

import
    Nabla

CUDA.allowscalar() do
    include("core.jl")

    @testset "Overall" begin
        include("smoke_tests.jl")
    end
end

using ICNF,
    AbstractDifferentiation,
    Aqua,
    CUDA,
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
    Zygote

CUDA.allowscalar() do
    include("core.jl")

    @testset "Overall" begin
        include("smoke_tests.jl")
    end

    # Aqua.test_all(ICNF)
end

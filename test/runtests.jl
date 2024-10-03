import ADTypes,
    Aqua,
    ComponentArrays,
    ComputationalResources,
    CUDA,
    cuDNN,
    DataFrames,
    DifferentiationInterface,
    Distances,
    Distributions,
    Enzyme,
    GPUArraysCore,
    JET,
    Logging,
    Lux,
    LuxCUDA,
    MLJBase,
    SciMLBase,
    StableRNGs,
    TerminalLoggers,
    Test,
    Zygote,
    ContinuousNormalizingFlows

GROUP = get(ENV, "GROUP", "All")
USE_GPU = get(ENV, "USE_GPU", "Yes") == "Yes"

if (GROUP == "All")
    GC.enable_logging(true)

    debuglogger = TerminalLoggers.TerminalLogger(stderr, Logging.Debug)
    Logging.global_logger(debuglogger)
end

Test.@testset "Overall" begin
    if GROUP == "All" ||
       GROUP in ["RNODE", "FFJORD", "Planar", "CondRNODE", "CondFFJORD", "CondPlanar"]
        GPUArraysCore.allowscalar() do
            include("smoke_tests.jl")
        end
    end

    if GROUP == "All" || GROUP == "Quality"
        include("quality_tests.jl")
    end

    if GROUP == "All" || GROUP == "Instability"
        include("instability_tests.jl")
    end

    if GROUP == "All" || GROUP == "Regression"
        include("regression_tests.jl")
    end
end

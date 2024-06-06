import AbstractDifferentiation,
    ADTypes,
    Aqua,
    ComponentArrays,
    ComputationalResources,
    CUDA,
    cuDNN,
    DataFrames,
    DifferentiationInterface,
    Distances,
    Distributions,
    ForwardDiff,
    JET,
    Logging,
    Lux,
    LuxCUDA,
    MLJBase,
    ReverseDiff,
    SciMLBase,
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
else
    warnlogger = TerminalLoggers.TerminalLogger(stderr, Logging.Warn)
    Logging.global_logger(warnlogger)
end

Test.@testset "Overall" begin
    if GROUP == "All" ||
       GROUP in ["RNODE", "FFJORD", "Planar", "CondRNODE", "CondFFJORD", "CondPlanar"]
        CUDA.allowscalar() do
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

import ADTypes,
    Aqua,
    ComponentArrays,
    DataFrames,
    DifferentiationInterface,
    Distances,
    Distributions,
    ForwardDiff,
    JET,
    Logging,
    Lux,
    MLDataDevices,
    MLJBase,
    StableRNGs,
    TerminalLoggers,
    Test,
    Zygote,
    ContinuousNormalizingFlows

GROUP = get(ENV, "GROUP", "All")

if (GROUP == "All")
    GC.enable_logging(true)

    debuglogger = TerminalLoggers.TerminalLogger(stderr, Logging.Debug)
    Logging.global_logger(debuglogger)
end

Test.@testset "Overall" begin
    if GROUP == "All" || GROUP == "Smoke"
        include("smoke_tests.jl")
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

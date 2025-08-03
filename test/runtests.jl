import ADTypes,
    Aqua,
    ComponentArrays,
    DataFrames,
    DifferentiationInterface,
    Distances,
    Distributions,
    ExplicitImports,
    ForwardDiff,
    JET,
    Logging,
    Lux,
    MLDataDevices,
    MLJBase,
    OrdinaryDiffEqTsit5,
    SciMLSensitivity,
    StableRNGs,
    Static,
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
    if GROUP == "All" || GROUP in ["SmokeXOut", "SmokeXIn", "SmokeXYOut", "SmokeXYIn"]
        include("smoke_tests.jl")
    end

    if GROUP == "All" || GROUP == "Regression"
        include("regression_tests.jl")
    end

    if GROUP == "All" || GROUP == "CheckByAqua"
        include("checkby_Aqua_tests.jl")
    end

    if GROUP == "All" || GROUP == "CheckByJET"
        include("checkby_JET_tests.jl")
    end

    if GROUP == "All" || GROUP == "CheckByExplicitImports"
        include("checkby_ExplicitImports_tests.jl")
    end
end

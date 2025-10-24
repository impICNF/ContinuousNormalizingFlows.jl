import ADTypes,
    Aqua,
    ComponentArrays,
    DataFrames,
    DifferentiationInterface,
    Distances,
    Distributions,
    Enzyme,
    ExplicitImports,
    ForwardDiff,
    JET,
    Lux,
    LuxCore,
    MLDataDevices,
    MLJBase,
    OrdinaryDiffEqDefault,
    SciMLSensitivity,
    StableRNGs,
    Test,
    Zygote,
    ContinuousNormalizingFlows

GROUP = get(ENV, "GROUP", "All")

Test.@testset verbose = true showtiming = true failfast = false "Overall" begin
    if GROUP == "All" || GROUP in ["SmokeXOut", "SmokeXIn", "SmokeXYOut", "SmokeXYIn"]
        include(joinpath("ci_tests", "smoke_tests.jl"))
    end

    if GROUP == "All" || GROUP == "Regression"
        include(joinpath("ci_tests", "regression_tests.jl"))
    end

    if GROUP == "All" || GROUP == "Speed"
        include(joinpath("ci_tests", "speed_tests.jl"))
    end

    if GROUP == "All" || GROUP == "CheckByAqua"
        include(joinpath("quality_tests", "checkby_Aqua_tests.jl"))
    end

    if GROUP == "All" || GROUP == "CheckByJET"
        include(joinpath("quality_tests", "checkby_JET_tests.jl"))
    end

    if GROUP == "All" || GROUP == "CheckByExplicitImports"
        include(joinpath("quality_tests", "checkby_ExplicitImports_tests.jl"))
    end
end

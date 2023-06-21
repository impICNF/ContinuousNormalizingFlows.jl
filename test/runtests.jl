using ContinuousNormalizingFlows
using Test
using AbstractDifferentiation: AbstractDifferentiation
using ADTypes: ADTypes
using Aqua: Aqua
using BenchmarkTools: BenchmarkTools
using Calculus: Calculus
using ComponentArrays: ComponentArrays
using ComputationalResources: ComputationalResources
using CUDA: CUDA
using DataFrames: DataFrames
using Distributions: Distributions
using Enzyme: Enzyme
using FiniteDiff: FiniteDiff
using FiniteDifferences: FiniteDifferences
using Flux: Flux
using ForwardDiff: ForwardDiff
using Logging: Logging
using Lux: Lux
using MLJBase: MLJBase
using ModelingToolkit: ModelingToolkit
using Optimization: Optimization
using Random: Random
using ReverseDiff: ReverseDiff
using SciMLBase: SciMLBase
using Tracker: Tracker
using Zygote: Zygote

debuglogger = Logging.ConsoleLogger(Logging.Debug)
Logging.global_logger(debuglogger)

include("core.jl")

GROUP = get(ENV, "GROUP", "All")
SMALL = get(ENV, "SMALL", "No") == "Yes"

@testset "Overall" begin
    CUDA.allowscalar() do
        include("smoke_tests.jl")
    end

    if GROUP == "All" || GROUP == "Quality"
        include("quality_tests.jl")
    end

    if GROUP == "All" || GROUP == "Benchmark"
        include("benchmark_tests.jl")
    end
end

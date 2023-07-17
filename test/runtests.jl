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
using cuDNN: cuDNN
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
using TruncatedStacktraces: TruncatedStacktraces
using Zygote: Zygote

GC.enable_logging(true)

debuglogger = Logging.ConsoleLogger(Logging.Debug)
Logging.global_logger(debuglogger)

TruncatedStacktraces.VERBOSE[] = true

include("core.jl")

GROUP = get(ENV, "GROUP", "All")
SMALL = get(ENV, "SMALL", "No") == "Yes"

@testset "Overall" begin
    if GROUP == "All" || GROUP in ["RNODE", "FFJORD", "Planar"]
        CUDA.allowscalar() do
            include("smoke_tests.jl")
        end
    end

    if GROUP == "All" || GROUP == "Quality"
        include("quality_tests.jl")
    end

    if GROUP == "All" || GROUP == "Benchmark"
        include("benchmark_tests.jl")
    end
end

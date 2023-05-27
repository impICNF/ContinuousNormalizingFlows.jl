using ContinuousNormalizingFlows
using Test
using AbstractDifferentiation: AbstractDifferentiation
using Aqua: Aqua
using CUDA: CUDA
using Calculus: Calculus
using ComponentArrays: ComponentArrays
using ComputationalResources: ComputationalResources
using DataFrames: DataFrames
using Distributions: Distributions
using FiniteDiff: FiniteDiff
using FiniteDifferences: FiniteDifferences
using Flux: Flux
using ForwardDiff: ForwardDiff
using Logging: Logging
using Lux: Lux
using MLJBase: MLJBase
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
SMALL = get(ENV, "SMALL", "0") == "1"

@testset "Overall" begin
    CUDA.allowscalar() do
        include("smoke_tests.jl")
    end
end

if GROUP == "All" || GROUP == "Quality"
    include("quality_tests.jl")
end

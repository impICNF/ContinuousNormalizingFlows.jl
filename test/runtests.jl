using ContinuousNormalizingFlows
using Test
using AbstractDifferentiation: AbstractDifferentiation
using ADTypes: ADTypes
using Aqua: Aqua
using ComponentArrays: ComponentArrays
using ComputationalResources: ComputationalResources
using CUDA: CUDA
using cuDNN: cuDNN
using DataFrames: DataFrames
using DifferentiationInterface: DifferentiationInterface
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using JET: JET
using Logging: Logging
using Lux: Lux
using LuxCUDA: LuxCUDA
using MLJBase: MLJBase
using ReverseDiff: ReverseDiff
using SciMLBase: SciMLBase
using TerminalLoggers: TerminalLoggers
using Zygote: Zygote

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

@testset "Overall" begin
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
end

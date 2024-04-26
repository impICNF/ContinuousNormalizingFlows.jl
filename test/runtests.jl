using ContinuousNormalizingFlows
using Test
import AbstractDifferentiation
import ADTypes
import Aqua
import ComponentArrays
import ComputationalResources
import CUDA
import cuDNN
import DataFrames
import DifferentiationInterface
import Distributions
import ForwardDiff
import JET
import Logging
import Lux
import LuxCUDA
import MLJBase
import ReverseDiff
import SciMLBase
import TerminalLoggers
import Zygote

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

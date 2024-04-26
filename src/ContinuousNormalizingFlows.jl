module ContinuousNormalizingFlows

import AbstractDifferentiation,
    ADTypes,
    Base.Iterators,
    ChainRulesCore,
    ComponentArrays,
    ComputationalResources,
    DataFrames,
    Dates,
    DifferentialEquations,
    DifferentiationInterface,
    Distributions,
    DistributionsAD,
    FillArrays,
    LinearAlgebra,
    Lux,
    LuxCore,
    MLJBase,
    MLJModelInterface,
    MLUtils,
    NNlib,
    Octavian,
    Optimisers,
    Optimization,
    OptimizationOptimisers,
    Random,
    ScientificTypes,
    SciMLBase,
    SciMLSensitivity,
    Statistics,
    Zygote

include(joinpath("layers", "cond_layer.jl"))
include(joinpath("layers", "planar_layer.jl"))
include(joinpath("layers", "mul_layer.jl"))

include("types.jl")

include("base_icnf.jl")

include("icnf.jl")

include("utils.jl")

include(joinpath("cores", "core.jl"))
include(joinpath("cores", "core_icnf.jl"))
include(joinpath("cores", "core_cond_icnf.jl"))

"""
Implementations of Infinitesimal Continuous Normalizing Flows Algorithms in Julia
"""
ContinuousNormalizingFlows

end

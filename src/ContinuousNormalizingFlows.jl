module ContinuousNormalizingFlows

using AbstractDifferentiation,
    ADTypes,
    Base.Iterators,
    ChainRulesCore,
    ComponentArrays,
    ComputationalResources,
    CUDA,
    cuDNN,
    DataFrames,
    Dates,
    DifferentialEquations,
    Distributions,
    DistributionsAD,
    FillArrays,
    Flux,
    LinearAlgebra,
    Lux,
    LuxCore,
    LuxCUDA,
    MLJBase,
    MLJModelInterface,
    MLUtils,
    NNlib,
    NNlibCUDA,
    Optimisers,
    Optimization,
    OptimizationOptimisers,
    PrecompileTools,
    ProgressMeter,
    Random,
    ScientificTypes,
    SciMLBase,
    SciMLSensitivity,
    SparseDiffTools,
    Statistics,
    Zygote

include("types.jl")

include("planar_layer.jl")
include("flux_compat.jl")

include("base.jl")
include("base_icnf.jl")
include("base_cond_icnf.jl")

include("core.jl")
include("core_icnf.jl")
include("core_cond_icnf.jl")

include("rnode.jl")
include("ffjord.jl")
include("planar.jl")

include("cond_rnode.jl")
include("cond_ffjord.jl")
include("cond_planar.jl")

include("utils.jl")

@static if isdefined(Base, :get_extension)
    include("precompile.jl")
end

"""
Implementations of Infinitesimal Continuous Normalizing Flows Algorithms in Julia
"""
ContinuousNormalizingFlows

end

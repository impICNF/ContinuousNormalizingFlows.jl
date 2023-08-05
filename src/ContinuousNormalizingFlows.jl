module ContinuousNormalizingFlows

using AbstractDifferentiation,
    ADTypes,
    Base.Iterators,
    ChainRulesCore,
    ComponentArrays,
    ComputationalResources,
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
    MLJBase,
    MLJModelInterface,
    MLUtils,
    NNlib,
    Optimisers,
    Optimization,
    OptimizationOptimisers,
    OrdinaryDiffEq,
    PackageExtensionCompat,
    PrecompileTools,
    Preferences,
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

dprcmpltn = @load_preference("doprecompilation", true)

@static if isdefined(Base, :get_extension) && dprcmpltn
    include("precompile.jl")
end

function __init__()
    @require_extensions
end

"""
Implementations of Infinitesimal Continuous Normalizing Flows Algorithms in Julia
"""
ContinuousNormalizingFlows

end

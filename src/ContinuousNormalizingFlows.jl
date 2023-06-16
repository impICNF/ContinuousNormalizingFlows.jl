module ContinuousNormalizingFlows

using ADTypes,
    AbstractDifferentiation,
    CUDA,
    ComponentArrays,
    ComputationalResources,
    DataFrames,
    DifferentialEquations,
    Distributions,
    DistributionsAD,
    FillArrays,
    Flux,
    IterTools,
    Lux,
    LuxCUDA,
    LuxCore,
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
    SciMLBase,
    SciMLSensitivity,
    ScientificTypes,
    SparseDiffTools,
    Zygote,
    Dates,
    LinearAlgebra,
    Random,
    Statistics,
    Base.Iterators

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

end

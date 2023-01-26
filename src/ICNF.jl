module ICNF

using Adapt,
    CUDA,
    DataFrames,
    Distributions,
    DistributionsAD,
    FillArrays,
    Flux,
    Functors,
    IterTools,
    LineSearches,
    MLJBase,
    MLJModelInterface,
    MLUtils,
    NNlib,
    NNlibCUDA,
    Optim,
    Optimisers,
    Optimization,
    OptimizationFlux,
    OptimizationOptimJL,
    OptimizationOptimisers,
    SciMLBase,
    ScientificTypes,
    Zygote,
    LinearAlgebra,
    Random,
    Statistics

include("types.jl")
include("defaults.jl")
include("base_icnf.jl")
include("base_cond_icnf.jl")
include("core_icnf.jl")
include("core_cond_icnf.jl")

include("rnode.jl")
include("ffjord.jl")
include("planar.jl")

include("cond_rnode.jl")
include("cond_ffjord.jl")
include("cond_planar.jl")

include("metrics.jl")

include("utils.jl")

end

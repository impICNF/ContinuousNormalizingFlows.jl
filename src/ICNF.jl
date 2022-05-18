module ICNF

    using
        CUDA,
        ComputationalResources,
        DataFrames,
        DiffEqSensitivity,
        DifferentialEquations,
        Distributions,
        DistributionsAD,
        Flux,
        GalacticFlux,
        GalacticOptim,
        GalacticOptimJL,
        IterTools,
        MLJBase,
        MLJFlux,
        MLJModelInterface,
        MLUtils,
        NNlib,
        Optim,
        Optimisers,
        OrdinaryDiffEq,
        SciMLBase,
        ScientificTypes,
        Zygote,
        LinearAlgebra,
        Random,
        Statistics

    include("types.jl")
    include("defaults.jl")
    include("core_icnf.jl")
    include("core_cond_icnf.jl")

    include("ffjord.jl")
    include("rnode.jl")
    include("planar.jl")

    include("cond_ffjord.jl")
    include("cond_rnode.jl")
    include("cond_planar.jl")

    include("metrics.jl")

    include("utils.jl")

end

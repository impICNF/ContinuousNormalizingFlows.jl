module ContinuousNormalizingFlows

import AbstractDifferentiation,
    ADTypes,
    Base.Iterators,
    ChainRulesCore,
    ComponentArrays,
    ComputationalResources,
    DataFrames,
    Dates,
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
    OrdinaryDiffEq,
    Random,
    ScientificTypesBase,
    SciMLBase,
    SciMLSensitivity,
    Statistics,
    Zygote

export construct,
    inference,
    generate,
    loss,
    ICNF,
    RNODE,
    CondRNODE,
    FFJORD,
    CondFFJORD,
    Planar,
    CondPlanar,
    TestMode,
    TrainMode,
    ADVecJacVectorMode,
    ADJacVecVectorMode,
    DIVecJacVectorMode,
    DIJacVecVectorMode,
    DIVecJacMatrixMode,
    DIJacVecMatrixMode,
    ICNFModel,
    CondICNFModel,
    CondLayer,
    PlanarLayer,
    MulLayer

include(joinpath("layers", "cond_layer.jl"))
include(joinpath("layers", "planar_layer.jl"))
include(joinpath("layers", "mul_layer.jl"))

include("types.jl")

include("base_icnf.jl")

include("icnf.jl")

include("utils.jl")

include(joinpath("exts", "mlj_ext", "core.jl"))
include(joinpath("exts", "mlj_ext", "core_icnf.jl"))
include(joinpath("exts", "mlj_ext", "core_cond_icnf.jl"))

include(joinpath("exts", "dist_ext", "core.jl"))
include(joinpath("exts", "dist_ext", "core_icnf.jl"))
include(joinpath("exts", "dist_ext", "core_cond_icnf.jl"))

"""
Implementations of Infinitesimal Continuous Normalizing Flows Algorithms in Julia
"""
ContinuousNormalizingFlows

end

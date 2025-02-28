module ContinuousNormalizingFlows

import ADTypes,
    Base.Iterators,
    ChainRulesCore,
    ComponentArrays,
    DataFrames,
    Dates,
    DifferentiationInterface,
    Distributions,
    DistributionsAD,
    FillArrays,
    LinearAlgebra,
    Lux,
    LuxCore,
    MLDataDevices,
    MLJBase,
    MLJModelInterface,
    MLUtils,
    NNlib,
    Optimisers,
    Optimization,
    OptimizationOptimisers,
    OrdinaryDiffEqDefault,
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
    DIVecJacVectorMode,
    DIJacVecVectorMode,
    DIVecJacMatrixMode,
    DIJacVecMatrixMode,
    LuxVecJacMatrixMode,
    LuxJacVecMatrixMode,
    ICNFModel,
    CondICNFModel,
    CondLayer,
    PlanarLayer

include(joinpath("layers", "cond_layer.jl"))
include(joinpath("layers", "planar_layer.jl"))

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

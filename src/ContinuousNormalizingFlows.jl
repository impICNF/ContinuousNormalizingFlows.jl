module ContinuousNormalizingFlows

import ADTypes,
    ChainRulesCore,
    ComponentArrays,
    DataFrames,
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
    SciMLBase,
    SciMLSensitivity,
    ScientificTypesBase,
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

include(joinpath("core", "types.jl"))
include(joinpath("core", "base_icnf.jl"))
include(joinpath("core", "icnf.jl"))
include(joinpath("core", "utils.jl"))

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

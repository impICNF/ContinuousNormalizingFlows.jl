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
    OptimizationOptimisers,
    OrdinaryDiffEqAdamsBashforthMoulton,
    Random,
    SciMLBase,
    SciMLSensitivity,
    ScientificTypesBase,
    Static,
    Statistics,
    WeightInitializers,
    Zygote

export inference,
    generate,
    loss,
    ICNF,
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

include("layers/cond_layer.jl")
include("layers/planar_layer.jl")

include("core/types.jl")
include("core/base_icnf.jl")
include("core/icnf.jl")
include("core/utils.jl")

include("exts/mlj_ext/core.jl")
include("exts/mlj_ext/core_icnf.jl")
include("exts/mlj_ext/core_cond_icnf.jl")

include("exts/dist_ext/core.jl")
include("exts/dist_ext/core_icnf.jl")
include("exts/dist_ext/core_cond_icnf.jl")

"""
Implementations of Infinitesimal Continuous Normalizing Flows Algorithms in Julia
"""
ContinuousNormalizingFlows

end

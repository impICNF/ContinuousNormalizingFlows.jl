module ContinuousNormalizingFlows

import ADTypes,
    ChainRulesCore,
    ComponentArrays,
    DataFrames,
    DiffEqBase,
    DifferentiationInterface,
    Distributions,
    DistributionsAD,
    FastBroadcast,
    FillArrays,
    LinearAlgebra,
    LoopVectorization,
    Lux,
    LuxCore,
    MLDataDevices,
    MLJBase,
    MLJModelInterface,
    MLUtils,
    NNlib,
    Octavian,
    Optimisers,
    OptimizationBase,
    OptimizationOptimisers,
    OrdinaryDiffEqAdamsBashforthMoulton,
    Polyester,
    Random,
    SciMLBase,
    SciMLSensitivity,
    ScientificTypesBase,
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

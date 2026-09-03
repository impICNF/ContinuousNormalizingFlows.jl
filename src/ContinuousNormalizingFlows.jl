module ContinuousNormalizingFlows

import ADTypes,
    ChainRulesCore,
    ComponentArrays,
    DataFrames,
    DifferentiationInterface,
    Distributions,
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
    SciMLLogging,
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

Bibliography:

[Chen, Ricky TQ, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. "Neural Ordinary Differential Equations." arXiv preprint arXiv:1806.07366 (2018).](https://arxiv.org/abs/1806.07366)

[Grathwohl, Will, Ricky TQ Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. "Ffjord: Free-form continuous dynamics for scalable reversible generative models." arXiv preprint arXiv:1810.01367 (2018).](https://arxiv.org/abs/1810.01367)

[Finlay, Chris, Jörn-Henrik Jacobsen, Levon Nurbekyan, and Adam M. Oberman. "How to train your neural ODE: the world of Jacobian and kinetic regularization." arXiv preprint arXiv:2002.02798 (2020).](https://arxiv.org/abs/2002.02798)

[Dupont, Emilien, Arnaud Doucet, and Yee Whye Teh. "Augmented Neural ODEs." arXiv preprint arXiv:1904.01681 (2019).](https://arxiv.org/abs/1904.01681)

[Ghosh, Arnab, Harkirat Singh Behl, Emilien Dupont, Philip HS Torr, and Vinay Namboodiri. "STEER: Simple Temporal Regularization For Neural ODEs." arXiv preprint arXiv:2006.10711 (2020).](https://arxiv.org/abs/2006.10711)
"""
ContinuousNormalizingFlows

end

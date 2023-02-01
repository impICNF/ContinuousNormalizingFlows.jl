export TestMode, TrainMode

abstract type AbstractICNF{T, AT} <: LuxCore.AbstractExplicitContainerLayer{(
    :nn,
)} where {T <: AbstractFloat, AT <: AbstractArray} end
abstract type AbstractCondICNF{T, AT} <: LuxCore.AbstractExplicitContainerLayer{(
    :nn,
)} where {T <: AbstractFloat, AT <: AbstractArray} end

abstract type Mode end
struct TestMode <: Mode end
struct TrainMode <: Mode end

# MLJ interface

abstract type MLJICNF <: MLJModelInterface.Unsupervised end

# Distributions interface

abstract type ICNFDistribution <: ContinuousMultivariateDistribution end

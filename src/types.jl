export
    TestMode, TrainMode,
    FluxOptApp, OptimOptApp, SciMLOptApp

abstract type Flows end
abstract type NormalizingFlows <: Flows end
abstract type ContinuousNormalizingFlows <: NormalizingFlows end
abstract type InfinitesimalContinuousNormalizingFlows <: ContinuousNormalizingFlows end

abstract type AbstractICNF{T} <: InfinitesimalContinuousNormalizingFlows where {T <: AbstractFloat} end
abstract type AbstractCondICNF{T} <: InfinitesimalContinuousNormalizingFlows where {T <: AbstractFloat} end

abstract type Mode end
struct TestMode <: Mode end
struct TrainMode <: Mode end

abstract type OptApp end
struct FluxOptApp <: OptApp end
struct OptimOptApp <: OptApp end
struct SciMLOptApp <: OptApp end

# MLJ interface

abstract type MLJICNF <: MLJModelInterface.Unsupervised end

# Distributions interface

abstract type ICNFDistribution <: ContinuousMultivariateDistribution end

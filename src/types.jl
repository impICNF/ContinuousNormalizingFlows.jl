export TestMode,
    TrainMode, ADVectorMode, ZygoteMatrixMode, SDVecJacMatrixMode, SDJacVecMatrixMode

abstract type Mode end
struct TestMode <: Mode end
struct TrainMode <: Mode end

abstract type ComputeMode end
abstract type VectorMode <: ComputeMode end
abstract type MatrixMode <: ComputeMode end
abstract type SDMatrixMode <: MatrixMode end
struct ADVectorMode <: VectorMode end
struct SDVecJacMatrixMode <: SDMatrixMode end
struct SDJacVecMatrixMode <: SDMatrixMode end
struct ZygoteMatrixMode <: MatrixMode end

abstract type AbstractFlows{T <: AbstractFloat, CM <: ComputeMode, AUGMENTED, STEER} <:
              LuxCore.AbstractExplicitContainerLayer{(:nn,)} end
abstract type AbstractICNF{T <: AbstractFloat, CM <: ComputeMode, AUGMENTED, STEER} <:
              AbstractFlows{T, CM, AUGMENTED, STEER} end
abstract type AbstractCondICNF{T <: AbstractFloat, CM <: ComputeMode, AUGMENTED, STEER} <:
              AbstractFlows{T, CM, AUGMENTED, STEER} end

# MLJ interface

abstract type MLJICNF <: MLJModelInterface.Unsupervised end

# Distributions interface

abstract type ICNFDistribution <: ContinuousMultivariateDistribution end

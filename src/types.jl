export TestMode,
    TrainMode,
    ADVecJacVectorMode,
    ADJacVecVectorMode,
    SDVecJacMatrixMode,
    SDJacVecMatrixMode,
    ZygoteVectorMode,
    ZygoteMatrixMode

abstract type Mode end
struct TestMode <: Mode end
struct TrainMode <: Mode end

abstract type ComputeMode end
abstract type VectorMode <: ComputeMode end
abstract type MatrixMode <: ComputeMode end

abstract type ADVectorMode <: VectorMode end
struct ADVecJacVectorMode <: ADVectorMode end
struct ADJacVecVectorMode <: ADVectorMode end

abstract type SDMatrixMode <: MatrixMode end
struct SDVecJacMatrixMode <: SDMatrixMode end
struct SDJacVecMatrixMode <: SDMatrixMode end

struct ZygoteVectorMode <: VectorMode end
struct ZygoteMatrixMode <: MatrixMode end

abstract type AbstractICNF{
    T <: AbstractFloat,
    CM <: ComputeMode,
    INPLACE,
    COND,
    AUGMENTED,
    STEER,
    NORM_Z_AUG,
} <: LuxCore.AbstractExplicitContainerLayer{(:nn,)} end

# MLJ interface

abstract type MLJICNF <: MLJModelInterface.Unsupervised end

# Distributions interface

abstract type ICNFDistribution <: ContinuousMultivariateDistribution end

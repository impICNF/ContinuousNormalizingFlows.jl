export TestMode,
    TrainMode,
    ADVecJacVectorMode,
    ADJacVecVectorMode,
    DIVecJacVectorMode,
    DIJacVecVectorMode,
    DIVecJacMatrixMode,
    DIJacVecMatrixMode

abstract type Mode end
struct TestMode <: Mode end
struct TrainMode <: Mode end

abstract type ComputeMode end
abstract type VectorMode <: ComputeMode end
abstract type MatrixMode <: ComputeMode end

abstract type ADVectorMode <: VectorMode end
struct ADVecJacVectorMode <: ADVectorMode end
struct ADJacVecVectorMode <: ADVectorMode end

abstract type DIVectorMode <: VectorMode end
struct DIVecJacVectorMode <: DIVectorMode end
struct DIJacVecVectorMode <: DIVectorMode end

abstract type DIMatrixMode <: MatrixMode end
struct DIVecJacMatrixMode <: DIMatrixMode end
struct DIJacVecMatrixMode <: DIMatrixMode end

Base.Base.@deprecate_binding SDVecJacMatrixMode DIVecJacMatrixMode true
Base.Base.@deprecate_binding SDJacVecMatrixMode DIJacVecMatrixMode true

Base.Base.@deprecate_binding ZygoteVectorMode DIVecJacVectorMode true
Base.Base.@deprecate_binding ZygoteMatrixMode DIVecJacMatrixMode true

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

abstract type MLJICNF{AICNF <: AbstractICNF} <: MLJModelInterface.Unsupervised end

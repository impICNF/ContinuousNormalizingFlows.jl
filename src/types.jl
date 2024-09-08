abstract type Mode end
struct TestMode <: Mode end
struct TrainMode <: Mode end

abstract type ComputeMode{ADBack} end
abstract type VectorMode{ADBack} <: ComputeMode{ADBack} end
abstract type MatrixMode{ADBack} <: ComputeMode{ADBack} end

abstract type ADVectorMode{ADBack} <: VectorMode{ADBack} end
struct ADVecJacVectorMode{ADBack <: AbstractDifferentiation.AbstractBackend} <:
       ADVectorMode{ADBack}
    adback::ADBack
end
struct ADJacVecVectorMode{ADBack <: AbstractDifferentiation.AbstractBackend} <:
       ADVectorMode{ADBack}
    adback::ADBack
end

abstract type DIVectorMode{ADBack} <: VectorMode{ADBack} end
struct DIVecJacVectorMode{ADBack <: ADTypes.AbstractADType} <: DIVectorMode{ADBack}
    adback::ADBack
end
struct DIJacVecVectorMode{ADBack <: ADTypes.AbstractADType} <: DIVectorMode{ADBack}
    adback::ADBack
end

abstract type DIMatrixMode{ADBack} <: MatrixMode{ADBack} end
struct DIVecJacMatrixMode{ADBack <: ADTypes.AbstractADType} <: DIMatrixMode{ADBack}
    adback::ADBack
end
struct DIJacVecMatrixMode{ADBack <: ADTypes.AbstractADType} <: DIMatrixMode{ADBack}
    adback::ADBack
end

abstract type AbstractICNF{
    T <: AbstractFloat,
    CM <: ComputeMode,
    INPLACE,
    COND,
    AUGMENTED,
    STEER,
    NORM_Z_AUG,
} <: LuxCore.AbstractLuxContainerLayer{(:nn,)} end

abstract type MLJICNF{AICNF <: AbstractICNF} <: MLJModelInterface.Unsupervised end

@inline function ContinuousNormalizingFlows.zeros_T_AT(
    ::CUDALibs,
    ::ContinuousNormalizingFlows.AbstractFlows{T},
    dims...,
) where {T <: AbstractFloat}
    CUDA.zeros(T, dims...)
end

@inline function ContinuousNormalizingFlows.rand_T_AT(
    ::CUDALibs,
    ::ContinuousNormalizingFlows.AbstractFlows{T},
    dims...,
) where {T <: AbstractFloat}
    CUDA.rand(T, dims...)
end

@inline function ContinuousNormalizingFlows.randn_T_AT(
    ::CUDALibs,
    ::ContinuousNormalizingFlows.AbstractFlows{T},
    dims...,
) where {T <: AbstractFloat}
    CUDA.randn(T, dims...)
end

@inline function ContinuousNormalizingFlows.rand_cstm_AT(
    ::CUDALibs,
    icnf::ContinuousNormalizingFlows.AbstractFlows,
    cstm::Any,
    dims...,
)
    rcm = ContinuousNormalizingFlows.rand_cstm_AT(CPU1(), icnf, cstm, dims...)
    isempty(dims) ? rcm : convert(CuArray, rcm)
end

@non_differentiable CUDA.zeros(::Any...)
@non_differentiable CUDA.rand(::Any...)
@non_differentiable CUDA.randn(::Any...)

@inline function rng_AT(::CUDALibs)
    CURAND.default_rng()
end

@inline function ContinuousNormalizingFlows.zeros_T_AT(
    ::CUDALibs,
    ::ContinuousNormalizingFlows.AbstractFlows{T},
    dims...,
) where {T <: AbstractFloat}
    CUDA.zeros(T, dims...)
end

@inline function ContinuousNormalizingFlows.rand_cstm_AT(
    ::CUDALibs,
    icnf::ContinuousNormalizingFlows.AbstractFlows,
    cstm::Any,
    dims...,
)
    convert(CuArray, ContinuousNormalizingFlows.rand_cstm_AT(CPU1(), icnf, cstm, dims...))
end

@non_differentiable CUDA.zeros(::Any...)

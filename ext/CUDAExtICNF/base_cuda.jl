@inline function ContinuousNormalizingFlows.zeros_T_AT(
    resource::CUDALibs,
    ::ContinuousNormalizingFlows.AbstractFlows{T},
    dims...,
) where {T <: AbstractFloat}
    CUDA.zeros(T, dims...)
end

@inline function ContinuousNormalizingFlows.rand_T_AT(
    resource::CUDALibs,
    ::ContinuousNormalizingFlows.AbstractFlows{T},
    rng::AbstractRNG = Random.default_rng(),
    dims...,
) where {T <: AbstractFloat}
    CUDA.rand(T, dims...)
end

@inline function ContinuousNormalizingFlows.randn_T_AT(
    resource::CUDALibs,
    ::ContinuousNormalizingFlows.AbstractFlows{T},
    rng::AbstractRNG = Random.default_rng(),
    dims...,
) where {T <: AbstractFloat}
    CUDA.randn(T, dims...)
end

@inline function ContinuousNormalizingFlows.rand_cstm_AT(
    resource::CUDALibs,
    icnf::ContinuousNormalizingFlows.AbstractFlows{T},
    cstm::Any,
    rng::AbstractRNG = Random.default_rng(),
    dims...,
) where {T <: AbstractFloat}
    convert(
        CuArray,
        ContinuousNormalizingFlows.rand_cstm_AT(CPU1(), icnf, cstm, rng, dims...),
    )
end

@non_differentiable CUDA.zeros(::Any...)
@non_differentiable CUDA.rand(::Any...)
@non_differentiable CUDA.randn(::Any...)

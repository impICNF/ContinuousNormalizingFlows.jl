module ContinuousNormalizingFlowsCUDAExt

import CUDA, ComputationalResources, ContinuousNormalizingFlows

@inline function ContinuousNormalizingFlows.rng_AT(::ComputationalResources.CUDALibs)
    CUDA.CURAND.default_rng()
end

@inline function ContinuousNormalizingFlows.base_AT(
    ::CUDA.CUDALibs,
    ::ContinuousNormalizingFlows.AbstractICNF{T},
    dims...,
) where {T <: AbstractFloat}
    CUDA.CuArray{T}(undef, dims...)
end

end

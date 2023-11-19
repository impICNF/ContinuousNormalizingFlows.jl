module ContinuousNormalizingFlowsCUDAExt

using ContinuousNormalizingFlows, CUDA
using ContinuousNormalizingFlows.ComputationalResources

@inline function ContinuousNormalizingFlows.rng_AT(::CUDALibs)
    CURAND.default_rng()
end

@inline function ContinuousNormalizingFlows.rand_cstm_AT(
    ::CUDALibs,
    icnf::ContinuousNormalizingFlows.AbstractFlows,
    cstm::Any,
    dims...,
)
    convert(CuArray, ContinuousNormalizingFlows.rand_cstm_AT(CPU1(), icnf, cstm, dims...))
end

end

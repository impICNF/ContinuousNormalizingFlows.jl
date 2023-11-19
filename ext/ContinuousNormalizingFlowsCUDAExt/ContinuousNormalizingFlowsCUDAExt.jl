module ContinuousNormalizingFlowsCUDAExt

using ContinuousNormalizingFlows, CUDA
using ContinuousNormalizingFlows.ComputationalResources

@inline function ContinuousNormalizingFlows.rng_AT(::CUDALibs)
    CURAND.default_rng()
end

end

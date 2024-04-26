abstract type ICNFDistribution{AICNF <: ContinuousNormalizingFlows.AbstractICNF} <:
              ContinuousMultivariateDistribution end

function Base.length(d::ICNFDistribution)
    d.m.nvars
end

function Base.eltype(
    ::ICNFDistribution{AICNF},
) where {AICNF <: ContinuousNormalizingFlows.AbstractICNF}
    first(AICNF.parameters)
end

function Base.broadcastable(d::ICNFDistribution)
    Ref(d)
end

export ICNFDist, CondICNFDist

abstract type ICNFDistribution{AICNF <: AbstractICNF} <:
              Distributions.ContinuousMultivariateDistribution end

function Base.length(d::ICNFDistribution)
    return d.icnf.nvars
end

function Base.eltype(::ICNFDistribution{AICNF}) where {AICNF <: AbstractICNF}
    return first(AICNF.parameters)
end

function Base.broadcastable(d::ICNFDistribution)
    return Ref(d)
end

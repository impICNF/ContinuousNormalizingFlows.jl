export ICNFDist, CondICNFDist

abstract type ICNFDistribution{AICNF <: AbstractICNF} <:
              Distributions.ContinuousMultivariateDistribution end

function Base.length(d::ICNFDistribution)
    return d.icnf.nvariables
end

function Base.eltype(::ICNFDistribution{AICNF}) where {AICNF <: AbstractICNF}
    return eltype(AICNF)
end

function Base.broadcastable(d::ICNFDistribution)
    return Ref(d)
end

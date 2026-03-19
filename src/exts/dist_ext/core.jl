export ICNFDist, CondICNFDist

abstract type ICNFDistribution{AICNF <: AbstractICNF} <:
              Distributions.ContinuousMultivariateDistribution end

function Base.length(d::ICNFDistribution)
    return d.icnf.nvariables
end

function Base.eltype(d::ICNFDistribution)
    return eltype(d.icnf)
end

function Base.broadcastable(d::ICNFDistribution)
    return Ref(d)
end

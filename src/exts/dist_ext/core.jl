export ICNFDist, CondICNFDist

abstract type ICNFDistribution{AICNF <: AbstractICNF} <:
              Distributions.ContinuousMultivariateDistribution end

function Base.length(d::ICNFDistribution)
    d.m.nvars
end

function Base.eltype(::ICNFDistribution{AICNF}) where {AICNF <: AbstractICNF}
    first(AICNF.parameters)
end

function Base.broadcastable(d::ICNFDistribution)
    Ref(d)
end

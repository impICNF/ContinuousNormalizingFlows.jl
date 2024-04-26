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

struct ICNFDist{AICNF <: AbstractICNF} <: ICNFDistribution{AICNF}
    m::AICNF
    mode::Mode
    ps::Any
    st::NamedTuple
end

function ICNFDist(mach::MLJBase.Machine{<:ICNFModel}, mode::Mode)
    (ps, st) = fitted_params(mach)
    ICNFDist(mach.model.m, mode, ps, st)
end

function Distributions._logpdf(d::ICNFDist, x::AbstractVector{<:Real})
    if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        first(inference(d.m, d.mode, x, d.ps, d.st))
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        first(Distributions._logpdf(d, hcat(x)))
    else
        error("Not Implemented")
    end
end

function Distributions._logpdf(d::ICNFDist, A::AbstractMatrix{<:Real})
    if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        Distributions._logpdf.(d, eachcol(A))
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        first(inference(d.m, d.mode, A, d.ps, d.st))
    else
        error("Not Implemented")
    end
end

function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::ICNFDist,
    x::AbstractVector{<:Real},
)
    if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        x .= generate(d.m, d.mode, d.ps, d.st)
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        x .= Distributions._rand!(rng, d, hcat(x))
    else
        error("Not Implemented")
    end
end
function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::ICNFDist,
    A::AbstractMatrix{<:Real},
)
    if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        A .= hcat(Distributions._rand!.(rng, d, eachcol(A))...)
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        A .= generate(d.m, d.mode, d.ps, d.st, size(A, 2))
    else
        error("Not Implemented")
    end
end

struct CondICNFDist{AICNF <: AbstractICNF} <: ICNFDistribution{AICNF}
    m::AICNF
    mode::Mode
    ys::AbstractVecOrMat{<:Real}
    ps::Any
    st::NamedTuple
end

function CondICNFDist(
    mach::MLJBase.Machine{<:CondICNFModel},
    mode::Mode,
    ys::AbstractVecOrMat{<:Real},
)
    (ps, st) = fitted_params(mach)
    CondICNFDist(mach.model.m, mode, ys, ps, st)
end

function Distributions._logpdf(d::CondICNFDist, x::AbstractVector{<:Real})
    if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        first(inference(d.m, d.mode, x, d.ys, d.ps, d.st))
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        first(Distributions._logpdf(d, hcat(x)))
    else
        error("Not Implemented")
    end
end
function Distributions._logpdf(d::CondICNFDist, A::AbstractMatrix{<:Real})
    if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        Distributions._logpdf.(d, eachcol(A))
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        first(inference(d.m, d.mode, A, d.ys[:, begin:size(A, 2)], d.ps, d.st))
    else
        error("Not Implemented")
    end
end
function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::CondICNFDist,
    x::AbstractVector{<:Real},
)
    if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        x .= generate(d.m, d.mode, d.ys, d.ps, d.st)
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        x .= Distributions._rand!(rng, d, hcat(x))
    else
        error("Not Implemented")
    end
end
function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::CondICNFDist,
    A::AbstractMatrix{<:Real},
)
    if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        A .= hcat(Distributions._rand!.(rng, d, eachcol(A))...)
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        A .= generate(d.m, d.mode, d.ys[:, begin:size(A, 2)], d.ps, d.st, size(A, 2))
    else
        error("Not Implemented")
    end
end

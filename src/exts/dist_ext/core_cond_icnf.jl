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
    (ps, st) = MLJModelInterface.fitted_params(mach)
    return CondICNFDist(mach.model.m, mode, ys, ps, st)
end

function Distributions._logpdf(
    d::CondICNFDist{<:AbstractICNF{<:AbstractFloat, <:VectorMode}},
    x::AbstractVector{<:Real},
)
    return first(inference(d.m, d.mode, x, d.ys, d.ps, d.st))
end

function Distributions._logpdf(
    d::CondICNFDist{<:AbstractICNF{<:AbstractFloat, <:MatrixMode}},
    x::AbstractVector{<:Real},
)
    @warn "to compute by matrices, data should be a matrix."
    return first(Distributions._logpdf(d, hcat(x)))
end

function Distributions._logpdf(
    d::CondICNFDist{<:AbstractICNF{<:AbstractFloat, <:VectorMode}},
    A::AbstractMatrix{<:Real},
)
    @warn "to compute by vectors, data should be a vector."
    return Distributions._logpdf.(d, collect(collect.(eachcol(A))))
end

function Distributions._logpdf(
    d::CondICNFDist{<:AbstractICNF{<:AbstractFloat, <:MatrixMode}},
    A::AbstractMatrix{<:Real},
)
    return first(inference(d.m, d.mode, A, d.ys[:, begin:size(A, 2)], d.ps, d.st))
end

function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::CondICNFDist{<:AbstractICNF{<:AbstractFloat, <:VectorMode}},
    x::AbstractVector{<:Real},
)
    return x .= generate(d.m, d.mode, d.ys, d.ps, d.st)
end

function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::CondICNFDist{<:AbstractICNF{<:AbstractFloat, <:MatrixMode}},
    x::AbstractVector{<:Real},
)
    @warn "to compute by matrices, data should be a matrix."
    return x .= Distributions._rand!(rng, d, hcat(x))
end

function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::CondICNFDist{<:AbstractICNF{<:AbstractFloat, <:VectorMode}},
    A::AbstractMatrix{<:Real},
)
    @warn "to compute by vectors, data should be a vector."
    return A .= hcat(Distributions._rand!.(rng, d, collect(collect.(eachcol(A))))...)
end

function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::CondICNFDist{<:AbstractICNF{<:AbstractFloat, <:MatrixMode}},
    A::AbstractMatrix{<:Real},
)
    return A .= generate(d.m, d.mode, d.ys[:, begin:size(A, 2)], d.ps, d.st, size(A, 2))
end

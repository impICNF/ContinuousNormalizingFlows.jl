struct ICNFDist{AICNF <: AbstractICNF} <: ICNFDistribution{AICNF}
    m::AICNF
    mode::Mode
    ps::Any
    st::NamedTuple
end

function ICNFDist(mach::MLJBase.Machine{<:ICNFModel}, mode::Mode)
    (ps, st) = MLJModelInterface.fitted_params(mach)
    return ICNFDist(mach.model.m, mode, ps, st)
end

function Distributions._logpdf(
    d::ICNFDist{<:AbstractICNF{<:AbstractFloat, <:VectorMode}},
    x::AbstractVector{<:Real},
)
    return first(inference(d.m, d.mode, x, d.ps, d.st))
end

function Distributions._logpdf(
    d::ICNFDist{<:AbstractICNF{<:AbstractFloat, <:MatrixMode}},
    x::AbstractVector{<:Real},
)
    @warn maxlog = 1 "to compute by matrices, data should be a matrix."
    return first(Distributions._logpdf(d, hcat(x)))
end

function Distributions._logpdf(
    d::ICNFDist{<:AbstractICNF{<:AbstractFloat, <:VectorMode}},
    A::AbstractMatrix{<:Real},
)
    @warn maxlog = 1 "to compute by vectors, data should be a vector."
    return Distributions._logpdf.(d, collect(collect.(eachcol(A))))
end

function Distributions._logpdf(
    d::ICNFDist{<:AbstractICNF{<:AbstractFloat, <:MatrixMode}},
    A::AbstractMatrix{<:Real},
)
    return first(inference(d.m, d.mode, A, d.ps, d.st))
end

function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::ICNFDist{<:AbstractICNF{<:AbstractFloat, <:VectorMode}},
    x::AbstractVector{<:Real},
)
    return x .= generate(d.m, d.mode, d.ps, d.st)
end

function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::ICNFDist{<:AbstractICNF{<:AbstractFloat, <:MatrixMode}},
    x::AbstractVector{<:Real},
)
    @warn maxlog = 1 "to compute by matrices, data should be a matrix."
    return x .= Distributions._rand!(rng, d, hcat(x))
end

function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::ICNFDist{<:AbstractICNF{<:AbstractFloat, <:VectorMode}},
    A::AbstractMatrix{<:Real},
)
    @warn maxlog = 1 "to compute by vectors, data should be a vector."
    return A .= hcat(Distributions._rand!.(rng, d, collect(collect.(eachcol(A))))...)
end

function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::ICNFDist{<:AbstractICNF{<:AbstractFloat, <:MatrixMode}},
    A::AbstractMatrix{<:Real},
)
    return A .= generate(d.m, d.mode, d.ps, d.st, size(A, 2))
end

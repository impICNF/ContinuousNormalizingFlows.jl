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

function Distributions._logpdf(d::CondICNFDist, x::AbstractVector{<:Real})
    return if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        first(inference(d.m, d.mode, x, d.ys, d.ps, d.st))
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        @warn "to compute by matrices, data should be a matrix."
        first(Distributions._logpdf(d, hcat(x)))
    else
        error("Not Implemented")
    end
end
function Distributions._logpdf(d::CondICNFDist, A::AbstractMatrix{<:Real})
    return if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        @warn "to compute by vectors, data should be a vector."
        Distributions._logpdf.(d, collect(collect.(eachcol(A))))
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
    return if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        x .= generate(d.m, d.mode, d.ys, d.ps, d.st)
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        @warn "to compute by matrices, data should be a matrix."
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
    return if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        @warn "to compute by vectors, data should be a vector."
        A .= hcat(Distributions._rand!.(rng, d, collect(collect.(eachcol(A))))...)
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        A .= generate(d.m, d.mode, d.ys[:, begin:size(A, 2)], d.ps, d.st, size(A, 2))
    else
        error("Not Implemented")
    end
end

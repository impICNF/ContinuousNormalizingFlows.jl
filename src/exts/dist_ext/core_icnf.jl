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

function Distributions._logpdf(d::ICNFDist, x::AbstractVector{<:Real})
    return if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        first(inference(d.m, d.mode, x, d.ps, d.st))
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        @warn "to compute by matrices, data should be a matrix."
        first(Distributions._logpdf(d, hcat(x)))
    else
        error("Not Implemented")
    end
end

function Distributions._logpdf(d::ICNFDist, A::AbstractMatrix{<:Real})
    return if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        @warn "to compute by vectors, data should be a vector."
        Distributions._logpdf.(d, collect(collect.(eachcol(A))))
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
    return if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        x .= generate(d.m, d.mode, d.ps, d.st)
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        @warn "to compute by matrices, data should be a matrix."
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
    return if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        @warn "to compute by vectors, data should be a vector."
        A .= hcat(Distributions._rand!.(rng, d, collect(collect.(eachcol(A))))...)
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        A .= generate(d.m, d.mode, d.ps, d.st, size(A, 2))
    else
        error("Not Implemented")
    end
end

struct CondICNFDist{AICNF <: ContinuousNormalizingFlows.AbstractICNF} <:
       ICNFDistribution{AICNF}
    m::AICNF
    mode::ContinuousNormalizingFlows.Mode
    ys::AbstractVecOrMat{<:Real}
    ps::Any
    st::NamedTuple
end

function CondICNFDist(
    mach::ContinuousNormalizingFlows.Machine{<:ContinuousNormalizingFlows.CondICNFModel},
    mode::ContinuousNormalizingFlows.Mode,
    ys::AbstractVecOrMat{<:Real},
)
    (ps, st) = ContinuousNormalizingFlows.fitted_params(mach)
    CondICNFDist(mach.model.m, mode, ys, ps, st)
end

function Distributions._logpdf(d::CondICNFDist, x::AbstractVector{<:Real})
    if d.m isa ContinuousNormalizingFlows.AbstractICNF{
        <:AbstractFloat,
        <:ContinuousNormalizingFlows.VectorMode,
    }
        first(ContinuousNormalizingFlows.inference(d.m, d.mode, x, d.ys, d.ps, d.st))
    elseif d.m isa ContinuousNormalizingFlows.AbstractICNF{
        <:AbstractFloat,
        <:ContinuousNormalizingFlows.MatrixMode,
    }
        first(Distributions._logpdf(d, hcat(x)))
    else
        error("Not Implemented")
    end
end
function Distributions._logpdf(d::CondICNFDist, A::AbstractMatrix{<:Real})
    if d.m isa ContinuousNormalizingFlows.AbstractICNF{
        <:AbstractFloat,
        <:ContinuousNormalizingFlows.VectorMode,
    }
        Distributions._logpdf.(d, eachcol(A))
    elseif d.m isa ContinuousNormalizingFlows.AbstractICNF{
        <:AbstractFloat,
        <:ContinuousNormalizingFlows.MatrixMode,
    }
        first(
            ContinuousNormalizingFlows.inference(
                d.m,
                d.mode,
                A,
                d.ys[:, begin:size(A, 2)],
                d.ps,
                d.st,
            ),
        )
    else
        error("Not Implemented")
    end
end
function Distributions._rand!(
    rng::ContinuousNormalizingFlows.Random.AbstractRNG,
    d::CondICNFDist,
    x::AbstractVector{<:Real},
)
    if d.m isa ContinuousNormalizingFlows.AbstractICNF{
        <:AbstractFloat,
        <:ContinuousNormalizingFlows.VectorMode,
    }
        x .= ContinuousNormalizingFlows.generate(d.m, d.mode, d.ys, d.ps, d.st)
    elseif d.m isa ContinuousNormalizingFlows.AbstractICNF{
        <:AbstractFloat,
        <:ContinuousNormalizingFlows.MatrixMode,
    }
        x .= Distributions._rand!(rng, d, hcat(x))
    else
        error("Not Implemented")
    end
end
function Distributions._rand!(
    rng::ContinuousNormalizingFlows.Random.AbstractRNG,
    d::CondICNFDist,
    A::AbstractMatrix{<:Real},
)
    if d.m isa ContinuousNormalizingFlows.AbstractICNF{
        <:AbstractFloat,
        <:ContinuousNormalizingFlows.VectorMode,
    }
        A .= hcat(Distributions._rand!.(rng, d, eachcol(A))...)
    elseif d.m isa ContinuousNormalizingFlows.AbstractICNF{
        <:AbstractFloat,
        <:ContinuousNormalizingFlows.MatrixMode,
    }
        A .= ContinuousNormalizingFlows.generate(
            d.m,
            d.mode,
            d.ys[:, begin:size(A, 2)],
            d.ps,
            d.st,
            size(A, 2),
        )
    else
        error("Not Implemented")
    end
end

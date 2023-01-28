export agg_loglikelihood

function agg_loglikelihood(
    icnf::AbstractICNF{T, AT},
    xs::AbstractMatrix{<:Real},
    p::AbstractVector{<:Real} = icnf.p;
    agg::Function = mean,
    rng::AbstractRNG = Random.default_rng(),
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x, = inference(icnf, TestMode(), xs, p; rng)
    agg(logp̂x)
end

function agg_loglikelihood(
    icnf::AbstractCondICNF{T, AT},
    xs::AbstractMatrix{<:Real},
    ys::AbstractMatrix{<:Real},
    p::AbstractVector{<:Real} = icnf.p;
    agg::Function = mean,
    rng::AbstractRNG = Random.default_rng(),
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x, = inference(icnf, TestMode(), xs, ys, p; rng)
    agg(logp̂x)
end

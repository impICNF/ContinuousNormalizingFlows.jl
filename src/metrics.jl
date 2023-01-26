export agg_loglikelihood

function agg_loglikelihood(
    icnf::AbstractICNF{T, AT},
    xs::AbstractMatrix,
    p::AbstractVector = icnf.p;
    agg::Function = mean,
    rng::AbstractRNG = Random.default_rng(),
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x, = inference(icnf, TestMode(), xs, p; rng)
    agg(logp̂x)
end

function agg_loglikelihood(
    icnf::AbstractCondICNF{T, AT},
    xs::AbstractMatrix,
    ys::AbstractMatrix,
    p::AbstractVector = icnf.p;
    agg::Function = mean,
    rng::AbstractRNG = Random.default_rng(),
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x, = inference(icnf, TestMode(), xs, ys, p; rng)
    agg(logp̂x)
end

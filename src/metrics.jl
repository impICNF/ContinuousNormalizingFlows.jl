export agg_loglikelihood

function agg_loglikelihood(icnf::AbstractICNF{T}, xs::AbstractMatrix{T}; agg::Function=mean)::T where {T <: AbstractFloat}
    logp̂x = inference(icnf, TestMode(), xs)
    agg(logp̂x)
end

function agg_loglikelihood(icnf::AbstractICNF{T}, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}; agg::Function=mean)::T where {T <: AbstractFloat}
    logp̂x = inference(icnf, TestMode(), xs, ys)
    agg(logp̂x)
end

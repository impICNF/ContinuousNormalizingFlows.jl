export agg_loglikelihood

function agg_loglikelihood(icnf::AbstractICNF{T, AT}, xs::AbstractMatrix; agg::Function=mean)::Number where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x = inference(icnf, TestMode(), xs)
    agg(logp̂x)
end

function agg_loglikelihood(icnf::AbstractCondICNF{T, AT}, xs::AbstractMatrix, ys::AbstractMatrix; agg::Function=mean)::Number where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x = inference(icnf, TestMode(), xs, ys)
    agg(logp̂x)
end

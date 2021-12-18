function loglikelihood(icnf::AbstractICNF{T}, xs::AbstractMatrix{T}; agg::Function=mean)::T where {T <: AbstractFloat}
    logp̂x = inference(icnf, TestMode(), xs)
    agg(logp̂x)
end

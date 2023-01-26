export CondRNODE

"""
Implementation of RNODE (Conditional Version)
"""
struct CondRNODE{T <: AbstractFloat, AT <: AbstractArray} <: AbstractCondICNF{T, AT}
    re::Optimisers.Restructure
    p::AbstractVector{T}

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    # trace_test
    # trace_train
end

function CondRNODE{T, AT}(
    nn,
    nvars::Integer,
    ;
    basedist::Distribution = MvNormal(Zeros{T}(nvars), one(T) * I),
    tspan::Tuple{T, T} = convert(Tuple{T, T}, default_tspan),
    rng::AbstractRNG = Random.default_rng(),
) where {T <: AbstractFloat, AT <: AbstractArray}
    nn = fmap(x -> adapt(T, x), nn)
    p, re = destructure(nn)
    CondRNODE{T, AT}(re, convert(AT{T}, p), nvars, basedist, tspan)
end

function augmented_f(
    icnf::CondRNODE{T, AT},
    mode::TestMode,
    n_batch::Integer,
    ys::AbstractMatrix;
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    function f_aug(u, p, t)
        m = Chain(x -> vcat(x, ys), icnf.re(p))
        z = u[1:(end - 1), :]
        ż, J = jacobian_batched(m, z, T, AT)
        l̇ = transpose(tr.(eachslice(J; dims = 3)))
        vcat(ż, -l̇)
    end
    f_aug
end

function augmented_f(
    icnf::CondRNODE{T, AT},
    mode::TrainMode,
    n_batch::Integer,
    ys::AbstractMatrix;
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    ϵ = convert(AT, randn(rng, T, icnf.nvars, n_batch))

    function f_aug(u, p, t)
        m = Chain(x -> vcat(x, ys), icnf.re(p))
        z = u[1:(end - 3), :]
        ż, back = Zygote.pullback(m, z)
        ϵJ = only(back(ϵ))
        l̇ = sum(ϵJ .* ϵ; dims = 1)
        Ė = transpose(norm.(eachcol(ż)))
        ṅ = transpose(norm.(eachcol(ϵJ)))
        vcat(ż, -l̇, Ė, ṅ)
    end
    f_aug
end

@functor CondRNODE (p,)

function loss(
    icnf::CondRNODE{T, AT},
    xs::AbstractMatrix,
    ys::AbstractMatrix,
    p::AbstractVector = icnf.p,
    λ₁::T = convert(T, 1e-2),
    λ₂::T = convert(T, 1e-2);
    agg::Function = mean,
    rng::AbstractRNG = Random.default_rng(),
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x, Ė, ṅ = inference(icnf, TrainMode(), xs, ys, p; rng)
    agg(-logp̂x + λ₁ * Ė + λ₂ * ṅ)
end

function n_augment(
    icnf::CondRNODE{T, AT},
    mode::TrainMode,
)::Integer where {T <: AbstractFloat, AT <: AbstractArray}
    2
end

export RNODE

"""
Implementation of RNODE from

[Finlay, Chris, Jörn-Henrik Jacobsen, Levon Nurbekyan, and Adam M. Oberman. "How to train your neural ODE: the world of Jacobian and kinetic regularization." arXiv preprint arXiv:2002.02798 (2020).](https://arxiv.org/abs/2002.02798)
"""
struct RNODE{T <: AbstractFloat, AT <: AbstractArray} <: AbstractICNF{T, AT}
    re::Optimisers.Restructure
    p::AbstractVector{T}

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    # trace_test
    # trace_train
end

function RNODE{T, AT}(
    nn,
    nvars::Integer,
    ;
    basedist::Distribution = MvNormal(Zeros{T}(nvars), one(T) * I),
    tspan::Tuple{T, T} = convert(Tuple{T, T}, default_tspan),
    rng::AbstractRNG = Random.default_rng(),
) where {T <: AbstractFloat, AT <: AbstractArray}
    nn = fmap(x -> adapt(T, x), nn)
    p, re = destructure(nn)
    RNODE{T, AT}(re, convert(AT{T}, p), nvars, basedist, tspan)
end

function augmented_f(
    icnf::RNODE{T, AT},
    mode::TestMode,
    n_batch::Integer;
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    function f_aug(u, p, t)
        m = icnf.re(p)
        z = u[1:(end - 1), :]
        ż, J = jacobian_batched(m, z, T, AT)
        l̇ = transpose(tr.(eachslice(J; dims = 3)))
        vcat(ż, -l̇)
    end
    f_aug
end

function augmented_f(
    icnf::RNODE{T, AT},
    mode::TrainMode,
    n_batch::Integer;
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    ϵ = convert(AT, randn(rng, T, icnf.nvars, n_batch))

    function f_aug(u, p, t)
        m = icnf.re(p)
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

@functor RNODE (p,)

function loss(
    icnf::RNODE{T, AT},
    xs::AbstractMatrix,
    p::AbstractVector = icnf.p,
    λ₁::T = convert(T, 1e-2),
    λ₂::T = convert(T, 1e-2);
    agg::Function = mean,
    rng::AbstractRNG = Random.default_rng(),
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x, Ė, ṅ = inference(icnf, TrainMode(), xs, p; rng)
    agg(-logp̂x + λ₁ * Ė + λ₂ * ṅ)
end

function n_augment(
    icnf::RNODE{T, AT},
    mode::TrainMode,
)::Integer where {T <: AbstractFloat, AT <: AbstractArray}
    2
end

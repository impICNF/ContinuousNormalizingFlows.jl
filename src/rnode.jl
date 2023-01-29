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
    mode::TestMode;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1

    function f_aug(u, p, t)
        m = icnf.re(p)
        z = u[1:(end - n_aug)]
        ż, J = AbstractDifferentiation.value_and_jacobian(differentiation_backend, m, z)
        l̇ = tr(only(J))
        vcat(ż, -l̇)
    end
    f_aug
end

function augmented_f(
    icnf::RNODE{T, AT},
    mode::TrainMode;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1
    ϵ = convert(AT, randn(rng, T, icnf.nvars))

    function f_aug(u, p, t)
        m = icnf.re(p)
        z = u[1:(end - n_aug)]
        v_pb = AbstractDifferentiation.value_and_pullback_function(
            differentiation_backend,
            m,
            z,
        )
        ż, ϵJ = v_pb(ϵ)
        ϵJ = only(ϵJ)
        l̇ = ϵJ ⋅ ϵ
        Ė = norm(ż)
        ṅ = norm(ϵJ)
        vcat(ż, -l̇, Ė, ṅ)
    end
    f_aug
end

@functor RNODE (p,)

function loss(
    icnf::RNODE{T, AT},
    xs::AbstractVector{<:Real},
    p::AbstractVector{<:Real} = icnf.p,
    λ₁::T = convert(T, 1e-2),
    λ₂::T = convert(T, 1e-2);
    rng::AbstractRNG = Random.default_rng(),
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x, Ė, ṅ = inference(icnf, TrainMode(), xs, p; rng)
    -logp̂x + λ₁ * Ė + λ₂ * ṅ
end

function n_augment(
    icnf::RNODE{T, AT},
    mode::TrainMode,
)::Integer where {T <: AbstractFloat, AT <: AbstractArray}
    2
end

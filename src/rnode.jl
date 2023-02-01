export RNODE

"""
Implementation of RNODE from

[Finlay, Chris, Jörn-Henrik Jacobsen, Levon Nurbekyan, and Adam M. Oberman. "How to train your neural ODE: the world of Jacobian and kinetic regularization." arXiv preprint arXiv:2002.02798 (2020).](https://arxiv.org/abs/2002.02798)
"""
struct RNODE{T <: AbstractFloat, AT <: AbstractArray} <: AbstractICNF{T, AT}
    nn::LuxCore.AbstractExplicitLayer

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    # trace_test
    # trace_train
end

function RNODE{T, AT}(
    nn::LuxCore.AbstractExplicitLayer,
    nvars::Integer,
    ;
    basedist::Distribution = MvNormal(Zeros{T}(nvars), one(T) * I),
    tspan::Tuple{T, T} = convert(Tuple{T, T}, default_tspan),
) where {T <: AbstractFloat, AT <: AbstractArray}
    RNODE{T, AT}(nn, nvars, basedist, tspan)
end

function augmented_f(
    icnf::RNODE{T, AT},
    mode::TestMode,
    st::Any;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1

    function f_aug(u, p, t)
        z = u[1:(end - n_aug)]
        ż, J = AbstractDifferentiation.value_and_jacobian(
            differentiation_backend,
            x -> first(LuxCore.apply(icnf.nn, x, p, st)),
            z,
        )
        l̇ = tr(only(J))
        vcat(ż, -l̇)
    end
    f_aug
end

function augmented_f(
    icnf::RNODE{T, AT},
    mode::TrainMode,
    st::Any;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1
    ϵ = convert(AT, randn(rng, T, icnf.nvars))

    function f_aug(u, p, t)
        z = u[1:(end - n_aug)]
        v_pb = AbstractDifferentiation.value_and_pullback_function(
            differentiation_backend,
            x -> first(LuxCore.apply(icnf.nn, x, p, st)),
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

function loss(
    icnf::RNODE{T, AT},
    xs::AbstractVector{<:Real},
    ps::Any,
    st::Any,
    λ₁::T = convert(T, 1e-2),
    λ₂::T = convert(T, 1e-2);
    rng::AbstractRNG = Random.default_rng(),
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x, Ė, ṅ = inference(icnf, TrainMode(), xs, ps, st; rng)
    -logp̂x + λ₁ * Ė + λ₂ * ṅ
end

function n_augment(
    icnf::RNODE{T, AT},
    mode::TrainMode,
)::Integer where {T <: AbstractFloat, AT <: AbstractArray}
    2
end

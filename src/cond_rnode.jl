export CondRNODE

"""
Implementation of RNODE (Conditional Version)
"""
struct CondRNODE{T <: AbstractFloat, AT <: AbstractArray} <: AbstractCondICNF{T, AT}
    nn::LuxCore.AbstractExplicitLayer

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    differentiation_backend::AbstractDifferentiation.AbstractBackend

    # trace_test
    # trace_train
end

function CondRNODE{T, AT}(
    nn::LuxCore.AbstractExplicitLayer,
    nvars::Integer,
    ;
    basedist::Distribution = MvNormal(Zeros{T}(nvars), one(T) * I),
    tspan::Tuple{T, T} = convert(Tuple{T, T}, (0, 1)),
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
) where {T <: AbstractFloat, AT <: AbstractArray}
    CondRNODE{T, AT}(nn, nvars, basedist, tspan, differentiation_backend)
end

function augmented_f(
    icnf::CondRNODE{T, AT},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    st::Any;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1

    function f_aug(u, p, t)
        z = u[1:(end - n_aug)]
        ż, J = AbstractDifferentiation.value_and_jacobian(
            differentiation_backend,
            x -> first(LuxCore.apply(icnf.nn, vcat(x, ys), p, st)),
            z,
        )
        l̇ = tr(only(J))
        vcat(ż, -l̇)
    end
    f_aug
end

function augmented_f(
    icnf::CondRNODE{T, AT},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
    st::Any;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1
    ϵ = convert(AT, randn(rng, T, icnf.nvars))

    function f_aug(u, p, t)
        z = u[1:(end - n_aug)]
        v_pb = AbstractDifferentiation.value_and_pullback_function(
            differentiation_backend,
            x -> first(LuxCore.apply(icnf.nn, vcat(x, ys), p, st)),
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
    icnf::CondRNODE{T, AT},
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any,
    λ₁::T = convert(T, 1e-2),
    λ₂::T = convert(T, 1e-2);
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    mode::Mode = TrainMode(),
    rng::AbstractRNG = Random.default_rng(),
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x, Ė, ṅ = inference(icnf, mode, xs, ys, ps, st; differentiation_backend, rng)
    -logp̂x + λ₁ * Ė + λ₂ * ṅ
end

function n_augment(
    icnf::CondRNODE{T, AT},
    mode::TrainMode,
)::Integer where {T <: AbstractFloat, AT <: AbstractArray}
    2
end

export FFJORD

"""
Implementation of FFJORD from

[Grathwohl, Will, Ricky TQ Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. "Ffjord: Free-form continuous dynamics for scalable reversible generative models." arXiv preprint arXiv:1810.01367 (2018).](https://arxiv.org/abs/1810.01367)
"""
struct FFJORD{T <: AbstractFloat, AT <: AbstractArray} <: AbstractICNF{T, AT}
    nn::LuxCore.AbstractExplicitLayer

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    # trace_test
    # trace_train
end

function FFJORD{T, AT}(
    nn::LuxCore.AbstractExplicitLayer,
    nvars::Integer,
    ;
    basedist::Distribution = MvNormal(Zeros{T}(nvars), one(T) * I),
    tspan::Tuple{T, T} = convert(Tuple{T, T}, default_tspan),
) where {T <: AbstractFloat, AT <: AbstractArray}
    FFJORD{T, AT}(nn, nvars, basedist, tspan)
end

function augmented_f(
    icnf::FFJORD{T, AT},
    mode::TestMode,
    st::Any;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1

    function f_aug(u, p, t)
        z = u[1:(end - n_aug)]
        mz, J = AbstractDifferentiation.value_and_jacobian(differentiation_backend, x -> first(LuxCore.apply(icnf.nn, x, p, st)), z)
        trace_J = tr(only(J))
        vcat(mz, -trace_J)
    end
    f_aug
end

function augmented_f(
    icnf::FFJORD{T, AT},
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
        mz, ϵJ = v_pb(ϵ)
        ϵJ = only(ϵJ)
        trace_J = ϵJ ⋅ ϵ
        vcat(mz, -trace_J)
    end
    f_aug
end

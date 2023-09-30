export FFJORD

"""
Implementation of FFJORD from

[Grathwohl, Will, Ricky TQ Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. "Ffjord: Free-form continuous dynamics for scalable reversible generative models." arXiv preprint arXiv:1810.01367 (2018).](https://arxiv.org/abs/1810.01367)
"""
struct FFJORD{
    T <: AbstractFloat,
    CM <: ComputeMode,
    INPLACE,
    AUGMENTED,
    STEER,
    NN <: LuxCore.AbstractExplicitLayer,
    NVARS <: Int,
    RESOURCE <: AbstractResource,
    BASEDIST <: Distribution,
    TSPAN <: NTuple{2, T},
    STEERDIST <: Distribution,
    DIFFERENTIATION_BACKEND <: AbstractDifferentiation.AbstractBackend,
    AUTODIFF_BACKEND <: ADTypes.AbstractADType,
    SOL_KWARGS <: Dict,
    RNG <: AbstractRNG,
} <: AbstractICNF{T, CM, INPLACE, AUGMENTED, STEER}
    nn::NN
    nvars::NVARS
    naugmented::NVARS

    resource::RESOURCE
    basedist::BASEDIST
    tspan::TSPAN
    steerdist::STEERDIST
    differentiation_backend::DIFFERENTIATION_BACKEND
    autodiff_backend::AUTODIFF_BACKEND
    sol_kwargs::SOL_KWARGS
    rng::RNG
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::FFJORD{T, <:ADVecJacVectorMode},
    mode::TrainMode,
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    v_pb = AbstractDifferentiation.value_and_pullback_function(
        icnf.differentiation_backend,
        let p = p, st = st
            x -> first(icnf.nn(x, p, st))
        end,
        z,
    )
    mz, ϵJ = v_pb(ϵ)
    ϵJ = only(ϵJ)
    trace_J = ϵJ ⋅ ϵ
    cat(mz, -trace_J; dims = 1)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::FFJORD{T, <:ADJacVecVectorMode},
    mode::TrainMode,
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    v_pb = AbstractDifferentiation.value_and_pushforward_function(
        icnf.differentiation_backend,
        let p = p, st = st
            x -> first(icnf.nn(x, p, st))
        end,
        z,
    )
    mz, Jϵ = v_pb(ϵ)
    Jϵ = only(Jϵ)
    trace_J = ϵ ⋅ Jϵ
    cat(mz, -trace_J; dims = 1)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::FFJORD{T, <:ZygoteVectorMode},
    mode::TrainMode,
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    mz, back = Zygote.pullback(let p = p, st = st
        x -> first(icnf.nn(x, p, st))
    end, z)
    ϵJ = only(back(ϵ))
    trace_J = ϵJ ⋅ ϵ
    cat(mz, -trace_J; dims = 1)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::FFJORD{T, <:SDVecJacMatrixMode},
    mode::TrainMode,
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    mz = first(icnf.nn(z, p, st))
    Jf = VecJac(let p = p, st = st
        x -> first(icnf.nn(x, p, st))
    end, z; autodiff = icnf.autodiff_backend)
    ϵJ = reshape(Jf * ϵ, size(z))
    trace_J = sum(ϵJ .* ϵ; dims = 1)
    cat(mz, -trace_J; dims = 1)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::FFJORD{T, <:SDJacVecMatrixMode},
    mode::TrainMode,
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    mz = first(icnf.nn(z, p, st))
    Jf = JacVec(let p = p, st = st
        x -> first(icnf.nn(x, p, st))
    end, z; autodiff = icnf.autodiff_backend)
    Jϵ = reshape(Jf * ϵ, size(z))
    trace_J = sum(ϵ .* Jϵ; dims = 1)
    cat(mz, -trace_J; dims = 1)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::FFJORD{T, <:ZygoteMatrixMode},
    mode::TrainMode,
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    mz, back = Zygote.pullback(let p = p, st = st
        x -> first(icnf.nn(x, p, st))
    end, z)
    ϵJ = only(back(ϵ))
    trace_J = sum(ϵJ .* ϵ; dims = 1)
    cat(mz, -trace_J; dims = 1)
end

@views function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::FFJORD{T, <:ZygoteMatrixMode, true},
    mode::TrainMode,
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    mz, back = Zygote.pullback(let p = p, st = st
        x -> first(icnf.nn(x, p, st))
    end, z)
    ϵJ = only(back(ϵ))
    du[begin:(end - n_aug - 1), :] .= mz
    du[(end - n_aug), :] .= -transpose(sum(ϵJ .* ϵ; dims = 1))
end

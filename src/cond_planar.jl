export CondPlanar

"""
Implementation of Planar (Conditional Version)
"""
struct CondPlanar{
    T <: AbstractFloat,
    CM <: ComputeMode,
    AUGMENTED,
    STEER,
    NN <: PlanarLayer,
    NVARS <: Int,
    RESOURCE <: AbstractResource,
    BASEDIST <: Distribution,
    TSPAN <: NTuple{2, T},
    STEERDIST <: Distribution,
    DIFFERENTIATION_BACKEND <: AbstractDifferentiation.AbstractBackend,
    AUTODIFF_BACKEND <: ADTypes.AbstractADType,
    SOL_ARGS <: Tuple,
    SOL_KWARGS <: Dict,
    RNG <: AbstractRNG,
    _FNN <: Function,
} <: AbstractCondICNF{T, CM, AUGMENTED, STEER}
    nn::NN
    nvars::NVARS
    naugmented::NVARS

    resource::RESOURCE
    basedist::BASEDIST
    tspan::TSPAN
    steerdist::STEERDIST
    differentiation_backend::DIFFERENTIATION_BACKEND
    autodiff_backend::AUTODIFF_BACKEND
    sol_args::SOL_ARGS
    sol_kwargs::SOL_KWARGS
    rng::RNG
    _fnn::_FNN
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{<:AbstractFloat, <:ADVectorMode},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{<:Real},
    st::Any,
)
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    mz = icnf._fnn(cat(z, ys; dims = 1), p, st)
    trace_J =
        p.u ⋅ transpose(
            only(
                AbstractDifferentiation.jacobian(
                    icnf.differentiation_backend,
                    x -> first(pl_h(icnf.nn, cat(x, ys; dims = 1), p, st)),
                    z,
                ),
            ),
        )
    cat(mz, -trace_J; dims = 1)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{<:AbstractFloat, <:ADVectorMode},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{<:Real},
    st::Any,
)
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    mz = icnf._fnn(cat(z, ys; dims = 1), p, st)
    trace_J =
        p.u ⋅ transpose(
            only(
                AbstractDifferentiation.jacobian(
                    icnf.differentiation_backend,
                    x -> first(pl_h(icnf.nn, cat(x, ys; dims = 1), p, st)),
                    z,
                ),
            ),
        )
    cat(mz, -trace_J; dims = 1)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{<:AbstractFloat, <:ZygoteMatrixMode},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{<:Real},
    st::Any,
)
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    mz, back = Zygote.pullback(icnf._fnn, cat(z, ys; dims = 1), p, st)
    ϵJ = first(back(ϵ))
    trace_J = sum(ϵJ .* ϵ; dims = 1)
    cat(mz, -trace_J; dims = 1)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{<:AbstractFloat, <:SDVecJacMatrixMode},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{<:Real},
    st::Any,
)
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    mz = icnf._fnn(cat(z, ys; dims = 1), p, st)
    Jf = VecJac(
        x -> icnf._fnn(cat(x, ys; dims = 1), p, st),
        z;
        autodiff = icnf.autodiff_backend,
    )
    ϵJ = reshape(Jf * ϵ, size(z))
    trace_J = sum(ϵJ .* ϵ; dims = 1)
    cat(mz, -trace_J; dims = 1)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{<:AbstractFloat, <:SDJacVecMatrixMode},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{<:Real},
    st::Any,
)
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    mz = icnf._fnn(cat(z, ys; dims = 1), p, st)
    Jf = JacVec(
        x -> icnf._fnn(cat(x, ys; dims = 1), p, st),
        z;
        autodiff = icnf.autodiff_backend,
    )
    Jϵ = reshape(Jf * ϵ, size(z))
    trace_J = sum(ϵ .* Jϵ; dims = 1)
    cat(mz, -trace_J; dims = 1)
end

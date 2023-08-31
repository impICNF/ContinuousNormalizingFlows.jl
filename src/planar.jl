export Planar

"""
Implementation of Planar Flows from

[Chen, Ricky TQ, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. "Neural Ordinary Differential Equations." arXiv preprint arXiv:1806.07366 (2018).](https://arxiv.org/abs/1806.07366)
"""
struct Planar{
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
} <: AbstractICNF{T, CM, AUGMENTED, STEER}
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
    icnf::Planar{<:AbstractFloat, <:ADVectorMode},
    mode::TestMode,
    ϵ::AbstractVector{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    mz = icnf._fnn(z, p, st)
    trace_J =
        p.u ⋅ transpose(
            only(
                AbstractDifferentiation.jacobian(
                    differentiation_backend,
                    x -> first(pl_h(icnf.nn, x, p, st)),
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
    icnf::Planar{<:AbstractFloat, <:ADVectorMode},
    mode::TrainMode,
    ϵ::AbstractVector{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    mz = icnf._fnn(z, p, st)
    trace_J =
        p.u ⋅ transpose(
            only(
                AbstractDifferentiation.jacobian(
                    differentiation_backend,
                    x -> first(pl_h(icnf.nn, x, p, st)),
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
    icnf::Planar{<:AbstractFloat, <:ZygoteMatrixMode},
    mode::TrainMode,
    ϵ::AbstractMatrix{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    mz, back = Zygote.pullback(icnf._fnn, z, p, st)
    ϵJ = first(back(ϵ))
    trace_J = sum(ϵJ .* ϵ; dims = 1)
    cat(mz, -trace_J; dims = 1)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::Planar{<:AbstractFloat, <:SDVecJacMatrixMode},
    mode::TrainMode,
    ϵ::AbstractMatrix{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    mz = icnf._fnn(z, p, st)
    Jf = VecJac(x -> icnf._fnn(x, p, st), z; autodiff = autodiff_backend)
    ϵJ = reshape(Jf * ϵ, size(z))
    trace_J = sum(ϵJ .* ϵ; dims = 1)
    cat(mz, -trace_J; dims = 1)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::Planar{<:AbstractFloat, <:SDJacVecMatrixMode},
    mode::TrainMode,
    ϵ::AbstractMatrix{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    mz = icnf._fnn(z, p, st)
    Jf = JacVec(x -> icnf._fnn(x, p, st), z; autodiff = autodiff_backend)
    Jϵ = reshape(Jf * ϵ, size(z))
    trace_J = sum(ϵ .* Jϵ; dims = 1)
    cat(mz, -trace_J; dims = 1)
end

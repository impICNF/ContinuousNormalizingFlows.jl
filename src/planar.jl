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
    RESOURCE <: AbstractResource,
    BASEDIST <: Distribution,
    TSPAN <: NTuple{2, T},
    STEERDIST <: Distribution,
    DIFFERENTIATION_BACKEND <: AbstractDifferentiation.AbstractBackend,
    _FNN <: ComposedFunction,
} <: AbstractICNF{T, CM, AUGMENTED, STEER}
    nn::NN
    nvars::Int
    naugmented::Int

    resource::RESOURCE
    basedist::BASEDIST
    tspan::TSPAN
    steerdist::STEERDIST
    differentiation_backend::DIFFERENTIATION_BACKEND
    sol_args::Tuple
    sol_kwargs::Dict
    _fnn::_FNN
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::Planar{<:AbstractFloat, <:ADVectorMode},
    mode::TestMode,
    ϵ::AbstractVector{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = @view u[begin:(end - n_aug - 1)]
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
    vcat(mz, -trace_J)
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::Planar{<:AbstractFloat, <:ADVectorMode},
    mode::TrainMode,
    ϵ::AbstractVector{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = @view u[begin:(end - n_aug - 1)]
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
    vcat(mz, -trace_J)
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::Planar{<:AbstractFloat, <:ZygoteMatrixMode},
    mode::TrainMode,
    ϵ::AbstractMatrix{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = @view u[begin:(end - n_aug - 1), :]
    mz, back = Zygote.pullback(icnf._fnn, z, p, st)
    ϵJ = first(back(ϵ))
    trace_J = sum(ϵJ .* ϵ; dims = 1)
    vcat(mz, -trace_J)
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::Planar{<:AbstractFloat, <:SDVecJacMatrixMode},
    mode::TrainMode,
    ϵ::AbstractMatrix{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = @view u[begin:(end - n_aug - 1), :]
    mz = icnf._fnn(z, p, st)
    ϵJ = reshape(auto_vecjac(x -> icnf._fnn(x, p, st), z, ϵ), size(z))
    trace_J = sum(ϵJ .* ϵ; dims = 1)
    vcat(mz, -trace_J)
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::Planar{<:AbstractFloat, <:SDJacVecMatrixMode},
    mode::TrainMode,
    ϵ::AbstractMatrix{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = @view u[begin:(end - n_aug - 1), :]
    mz = icnf._fnn(z, p, st)
    Jϵ = reshape(auto_jacvec(x -> icnf._fnn(x, p, st), z, ϵ), size(z))
    trace_J = sum(ϵ .* Jϵ; dims = 1)
    vcat(mz, -trace_J)
end

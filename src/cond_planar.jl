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
    RESOURCE <: AbstractResource,
    BASEDIST <: Distribution,
    TSPAN <: NTuple{2, T},
    STEERDIST <: Distribution,
    DIFFERENTIATION_BACKEND <: AbstractDifferentiation.AbstractBackend,
    _FNN <: Function,
} <: AbstractCondICNF{T, CM, AUGMENTED, STEER}
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
    icnf::CondPlanar{<:AbstractFloat, <:ADVectorMode},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = @view u[begin:(end - n_aug - 1)]
    mz = icnf._fnn([z; ys], p, st)
    trace_J =
        p.u ⋅ transpose(
            only(
                AbstractDifferentiation.jacobian(
                    differentiation_backend,
                    x -> first(pl_h(icnf.nn, [x; ys], p, st)),
                    z,
                ),
            ),
        )
    [mz; -trace_J]
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{<:AbstractFloat, <:ADVectorMode},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = @view u[begin:(end - n_aug - 1)]
    mz = icnf._fnn([z; ys], p, st)
    trace_J =
        p.u ⋅ transpose(
            only(
                AbstractDifferentiation.jacobian(
                    differentiation_backend,
                    x -> first(pl_h(icnf.nn, [x; ys], p, st)),
                    z,
                ),
            ),
        )
    [mz; -trace_J]
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{<:AbstractFloat, <:ZygoteMatrixMode},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = @view u[begin:(end - n_aug - 1), :]
    mz, back = Zygote.pullback(icnf._fnn, [z; ys], p, st)
    ϵJ = first(back(ϵ))
    trace_J = sum(ϵJ .* ϵ; dims = 1)
    [mz; -trace_J]
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{<:AbstractFloat, <:SDVecJacMatrixMode},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = @view u[begin:(end - n_aug - 1), :]
    mz = icnf._fnn([z; ys], p, st)
    ϵJ = reshape(auto_vecjac(x -> icnf._fnn([x; ys], p, st), z, ϵ), size(z))
    trace_J = sum(ϵJ .* ϵ; dims = 1)
    [mz; -trace_J]
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{<:AbstractFloat, <:SDJacVecMatrixMode},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = @view u[begin:(end - n_aug - 1), :]
    mz = icnf._fnn([z; ys], p, st)
    Jϵ = reshape(auto_jacvec(x -> icnf._fnn([x; ys], p, st), z, ϵ), size(z))
    trace_J = sum(ϵ .* Jϵ; dims = 1)
    [mz; -trace_J]
end

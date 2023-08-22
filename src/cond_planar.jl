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
    mz, _ = LuxCore.apply(icnf.nn, vcat(z, ys), p, st)
    trace_J =
        p.u ⋅ transpose(
            only(
                AbstractDifferentiation.jacobian(
                    differentiation_backend,
                    x -> first(pl_h(icnf.nn, vcat(x, ys), p, st)),
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
    mz, _ = LuxCore.apply(icnf.nn, vcat(z, ys), p, st)
    trace_J =
        p.u ⋅ transpose(
            only(
                AbstractDifferentiation.jacobian(
                    differentiation_backend,
                    x -> first(pl_h(icnf.nn, vcat(x, ys), p, st)),
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
    mz, back = Zygote.pullback(x -> first(LuxCore.apply(icnf.nn, vcat(x, ys), p, st)), z)
    ϵJ = only(back(ϵ))
    trace_J = sum(ϵJ .* ϵ; dims = 1)
    vcat(mz, -trace_J)
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
    mz = first(LuxCore.apply(icnf.nn, vcat(z, ys), p, st))
    ϵJ = reshape(
        auto_vecjac(x -> first(LuxCore.apply(icnf.nn, vcat(x, ys), p, st)), z, ϵ),
        size(z),
    )
    trace_J = sum(ϵJ .* ϵ; dims = 1)
    vcat(mz, -trace_J)
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
    mz = first(LuxCore.apply(icnf.nn, vcat(z, ys), p, st))
    Jϵ = reshape(
        auto_jacvec(x -> first(LuxCore.apply(icnf.nn, vcat(x, ys), p, st)), z, ϵ),
        size(z),
    )
    trace_J = sum(ϵ .* Jϵ; dims = 1)
    vcat(mz, -trace_J)
end

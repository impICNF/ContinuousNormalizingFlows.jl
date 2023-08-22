export CondFFJORD

"""
Implementation of FFJORD (Conditional Version)
"""
struct CondFFJORD{
    T <: AbstractFloat,
    CM <: ComputeMode,
    AUGMENTED,
    STEER,
    NN <: LuxCore.AbstractExplicitLayer,
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
    icnf::CondFFJORD{<:AbstractFloat, <:ADVectorMode},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    fnn = first ∘ icnf.nn
    z = @view u[begin:(end - n_aug - 1)]
    v_pb = AbstractDifferentiation.value_and_pullback_function(
        differentiation_backend,
        x -> fnn(vcat(x, ys), p, st),
        z,
    )
    mz, ϵJ = v_pb(ϵ)
    ϵJ = only(ϵJ)
    trace_J = ϵJ ⋅ ϵ
    vcat(mz, -trace_J)
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondFFJORD{<:AbstractFloat, <:ZygoteMatrixMode},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    fnn = first ∘ icnf.nn
    z = @view u[begin:(end - n_aug - 1), :]
    mz, back = Zygote.pullback(fnn, vcat(z, ys), p, st)
    ϵJ = first(back(ϵ))
    trace_J = sum(ϵJ .* ϵ; dims = 1)
    vcat(mz, -trace_J)
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondFFJORD{<:AbstractFloat, <:SDVecJacMatrixMode},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    fnn = first ∘ icnf.nn
    z = @view u[begin:(end - n_aug - 1), :]
    mz = fnn(vcat(z, ys), p, st)
    ϵJ = reshape(auto_vecjac(x -> fnn(vcat(x, ys), p, st), z, ϵ), size(z))
    trace_J = sum(ϵJ .* ϵ; dims = 1)
    vcat(mz, -trace_J)
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondFFJORD{<:AbstractFloat, <:SDJacVecMatrixMode},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    fnn = first ∘ icnf.nn
    z = @view u[begin:(end - n_aug - 1), :]
    mz = fnn(vcat(z, ys), p, st)
    Jϵ = reshape(auto_jacvec(x -> fnn(vcat(x, ys), p, st), z, ϵ), size(z))
    trace_J = sum(ϵ .* Jϵ; dims = 1)
    vcat(mz, -trace_J)
end

export CondPlanar

"""
Implementation of Planar (Conditional Version)
"""
struct CondPlanar{T <: AbstractFloat, CM <: ComputeMode, AUGMENTED, STEER} <:
       AbstractCondICNF{T, CM, AUGMENTED, STEER}
    nn::PlanarLayer
    nvars::Integer
    naugmented::Integer

    resource::AbstractResource
    basedist::Distribution
    tspan::NTuple{2, T}
    steer_rate::T
    differentiation_backend::AbstractDifferentiation.AbstractBackend
    sol_args::Tuple
    sol_kwargs::Dict
end

function augmented_f(
    u,
    p,
    t,
    icnf::CondPlanar{<:AbstractFloat, <:ADVectorMode},
    mode::TestMode,
    ys::AbstractVector{<:Real},
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
    u,
    p,
    t,
    icnf::CondPlanar{<:AbstractFloat, <:ADVectorMode},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
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
    u,
    p,
    t,
    icnf::CondPlanar{<:AbstractFloat, <:ZygoteMatrixMode},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    st::Any,
    n_batch::Integer;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    ϵ = randn_T_AT(resource, icnf, rng, icnf.nvars + n_aug_input, n_batch)
    z = @view u[begin:(end - n_aug - 1), :]
    mz, back = Zygote.pullback(x -> first(LuxCore.apply(icnf.nn, vcat(x, ys), p, st)), z)
    ϵJ = only(back(ϵ))
    trace_J = sum(ϵJ .* ϵ; dims = 1)
    vcat(mz, -trace_J)
end

function augmented_f(
    u,
    p,
    t,
    icnf::CondPlanar{<:AbstractFloat, <:SDVecJacMatrixMode},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    st::Any,
    n_batch::Integer;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    ϵ = randn_T_AT(resource, icnf, rng, icnf.nvars + n_aug_input, n_batch)
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
    u,
    p,
    t,
    icnf::CondPlanar{<:AbstractFloat, <:SDJacVecMatrixMode},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    st::Any,
    n_batch::Integer;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    ϵ = randn_T_AT(resource, icnf, rng, icnf.nvars + n_aug_input, n_batch)
    z = @view u[begin:(end - n_aug - 1), :]
    mz = first(LuxCore.apply(icnf.nn, vcat(z, ys), p, st))
    Jϵ = reshape(
        auto_jacvec(x -> first(LuxCore.apply(icnf.nn, vcat(x, ys), p, st)), z, ϵ),
        size(z),
    )
    trace_J = sum(ϵ .* Jϵ; dims = 1)
    vcat(mz, -trace_J)
end

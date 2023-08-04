export Planar

"""
Implementation of Planar Flows from

[Chen, Ricky TQ, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. "Neural Ordinary Differential Equations." arXiv preprint arXiv:1806.07366 (2018).](https://arxiv.org/abs/1806.07366)
"""
struct Planar{T <: AbstractFloat, CM <: ComputeMode, AUGMENTED, STEER} <:
       AbstractICNF{T, CM, AUGMENTED, STEER}
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
    icnf::Planar{<:AbstractFloat, <:ADVectorMode},
    mode::TestMode,
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = @view u[begin:(end - n_aug - 1)]
    mz, _ = LuxCore.apply(icnf.nn, z, p, st)
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
    u,
    p,
    t,
    icnf::Planar{<:AbstractFloat, <:ADVectorMode},
    mode::TrainMode,
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = @view u[begin:(end - n_aug - 1)]
    mz, _ = LuxCore.apply(icnf.nn, z, p, st)
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
    u,
    p,
    t,
    icnf::Planar{<:AbstractFloat, <:ZygoteMatrixMode},
    mode::TrainMode,
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
    mz, back = Zygote.pullback(x -> first(LuxCore.apply(icnf.nn, x, p, st)), z)
    ϵJ = only(back(ϵ))
    trace_J = sum(ϵJ .* ϵ; dims = 1)
    vcat(mz, -trace_J)
end

function augmented_f(
    u,
    p,
    t,
    icnf::Planar{<:AbstractFloat, <:SDVecJacMatrixMode},
    mode::TrainMode,
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
    mz = first(LuxCore.apply(icnf.nn, z, p, st))
    ϵJ = reshape(auto_vecjac(x -> first(LuxCore.apply(icnf.nn, x, p, st)), z, ϵ), size(z))
    trace_J = sum(ϵJ .* ϵ; dims = 1)
    vcat(mz, -trace_J)
end

function augmented_f(
    u,
    p,
    t,
    icnf::Planar{<:AbstractFloat, <:SDJacVecMatrixMode},
    mode::TrainMode,
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
    mz = first(LuxCore.apply(icnf.nn, z, p, st))
    Jϵ = reshape(auto_jacvec(x -> first(LuxCore.apply(icnf.nn, x, p, st)), z, ϵ), size(z))
    trace_J = sum(ϵ .* Jϵ; dims = 1)
    vcat(mz, -trace_J)
end

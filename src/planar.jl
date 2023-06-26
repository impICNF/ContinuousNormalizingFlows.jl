export Planar

"""
Implementation of Planar Flows from

[Chen, Ricky TQ, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. "Neural Ordinary Differential Equations." arXiv preprint arXiv:1806.07366 (2018).](https://arxiv.org/abs/1806.07366)
"""
struct Planar{T <: AbstractFloat, AT <: AbstractArray, CM <: ComputeMode} <:
       AbstractICNF{T, AT, CM}
    nn::PlanarLayer

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    differentiation_backend::AbstractDifferentiation.AbstractBackend

    sol_args::Tuple
    sol_kwargs::Dict

    # trace_test
    # trace_train
end

function augmented_f(
    icnf::Planar{T, AT, <:ADVectorMode},
    mode::Mode,
    st::Any;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1

    function f_aug(u, p, t)
        z = @view u[begin:(end - n_aug)]
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
    f_aug
end

function augmented_f(
    icnf::Planar{T, AT, CM},
    mode::TestMode,
    st::Any,
    n_batch::Integer;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray, CM <: MatrixMode}
    n_aug = n_augment(icnf, mode) + 1

    function f_aug(u, p, t)
        z = @view u[begin:(end - n_aug), :]
        mz, J = jacobian_batched(x -> first(LuxCore.apply(icnf.nn, x, p, st)), z, T, AT, CM)
        trace_J = transpose(tr.(eachslice(J; dims = 3)))
        vcat(mz, -trace_J)
    end
    f_aug
end

function augmented_f(
    icnf::Planar{T, AT, <:ZygoteMatrixMode},
    mode::TrainMode,
    st::Any,
    n_batch::Integer;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1
    ϵ::AT = randn(rng, T, icnf.nvars, n_batch)

    function f_aug(u, p, t)
        z = @view u[begin:(end - n_aug), :]
        mz, back = Zygote.pullback(x -> first(LuxCore.apply(icnf.nn, x, p, st)), z)
        ϵJ = only(back(ϵ))
        trace_J = sum(ϵJ .* ϵ; dims = 1)
        vcat(mz, -trace_J)
    end
    f_aug
end

function augmented_f(
    icnf::Planar{T, AT, <:SDVecJacMatrixMode},
    mode::TrainMode,
    st::Any,
    n_batch::Integer;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1
    ϵ::AT = randn(rng, T, icnf.nvars, n_batch)

    function f_aug(u, p, t)
        z = @view u[begin:(end - n_aug), :]
        mz = first(LuxCore.apply(icnf.nn, z, p, st))
        ϵJ = reshape(
            auto_vecjac(x -> first(LuxCore.apply(icnf.nn, x, p, st)), z, ϵ),
            size(z),
        )
        trace_J = sum(ϵJ .* ϵ; dims = 1)
        vcat(mz, -trace_J)
    end
    f_aug
end

function augmented_f(
    icnf::Planar{T, AT, <:SDJacVecMatrixMode},
    mode::TrainMode,
    st::Any,
    n_batch::Integer;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1
    ϵ::AT = randn(rng, T, icnf.nvars, n_batch)

    function f_aug(u, p, t)
        z = @view u[begin:(end - n_aug), :]
        mz = first(LuxCore.apply(icnf.nn, z, p, st))
        Jϵ = reshape(
            auto_jacvec(x -> first(LuxCore.apply(icnf.nn, x, p, st)), z, ϵ),
            size(z),
        )
        trace_J = sum(ϵ .* Jϵ; dims = 1)
        vcat(mz, -trace_J)
    end
    f_aug
end

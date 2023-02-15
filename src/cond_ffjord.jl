export CondFFJORD

"""
Implementation of FFJORD (Conditional Version)
"""
struct CondFFJORD{T <: AbstractFloat, AT <: AbstractArray, CM <: ComputeMode} <: AbstractCondICNF{T, AT, CM}
    nn::LuxCore.AbstractExplicitLayer

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    differentiation_backend::AbstractDifferentiation.AbstractBackend

    # trace_test
    # trace_train
end

function augmented_f(
    icnf::CondFFJORD{T, AT, <: ADVectorMode},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    st::Any;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1

    function f_aug(u, p, t)
        z = u[1:(end - n_aug)]
        mz, J = AbstractDifferentiation.value_and_jacobian(
            differentiation_backend,
            x -> first(LuxCore.apply(icnf.nn, vcat(x, ys), p, st)),
            z,
        )
        trace_J = tr(only(J))
        vcat(mz, -trace_J)
    end
    f_aug
end

function augmented_f(
    icnf::CondFFJORD{T, AT, <: ZygoteMatrixMode},
    mode::TestMode,
    ys::AbstractMatrix{<:Real},
    st::Any,
    n_batch::Integer;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1

    function f_aug(u, p, t)
        z = u[1:(end - n_aug), :]
        mz, J = jacobian_batched(x -> first(LuxCore.apply(icnf.nn, vcat(x, ys), p, st)), z, T, AT)
        trace_J = transpose(tr.(eachslice(J; dims = 3)))
        vcat(mz, -trace_J)
    end
    f_aug
end

function augmented_f(
    icnf::CondFFJORD{T, AT, <: ADVectorMode},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
    st::Any;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1
    ϵ = convert(AT, randn(rng, T, icnf.nvars))

    function f_aug(u, p, t)
        z = u[1:(end - 1)]
        v_pb = AbstractDifferentiation.value_and_pullback_function(
            differentiation_backend,
            x -> first(LuxCore.apply(icnf.nn, vcat(x, ys), p, st)),
            z,
        )
        mz, ϵJ = v_pb(ϵ)
        ϵJ = only(ϵJ)
        trace_J = ϵJ ⋅ ϵ
        vcat(mz, -trace_J)
    end
    f_aug
end

function augmented_f(
    icnf::CondFFJORD{T, AT, <: ZygoteMatrixMode},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    st::Any,
    n_batch::Integer;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1
    ϵ = convert(AT, randn(rng, T, icnf.nvars, n_batch))

    function f_aug(u, p, t)
        z = u[1:(end - n_aug), :]
        mz, back = Zygote.pullback(x -> first(LuxCore.apply(icnf.nn, vcat(x, ys), p, st)), z)
        ϵJ = only(back(ϵ))
        trace_J = sum(ϵJ .* ϵ; dims = 1)
        vcat(mz, -trace_J)
    end
    f_aug
end

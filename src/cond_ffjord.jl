export CondFFJORD

"""
Implementation of FFJORD (Conditional Version)
"""
struct CondFFJORD{T <: AbstractFloat, AT <: AbstractArray} <: AbstractCondICNF{T, AT}
    re::Optimisers.Restructure
    p::AbstractVector{T}

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    # trace_test
    # trace_train
end

function CondFFJORD{T, AT}(
    nn,
    nvars::Integer,
    ;
    basedist::Distribution = MvNormal(Zeros{T}(nvars), one(T) * I),
    tspan::Tuple{T, T} = convert(Tuple{T, T}, default_tspan),
    rng::AbstractRNG = Random.default_rng(),
) where {T <: AbstractFloat, AT <: AbstractArray}
    nn = fmap(x -> adapt(T, x), nn)
    p, re = destructure(nn)
    CondFFJORD{T, AT}(re, convert(AT{T}, p), nvars, basedist, tspan)
end

function augmented_f(
    icnf::CondFFJORD{T, AT},
    mode::TestMode,
    ys::AbstractVector{<:Real};
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1

    function f_aug(u, p, t)
        m = Chain(x -> vcat(x, ys), icnf.re(p))
        z = u[1:(end - n_aug)]
        mz, J = AbstractDifferentiation.value_and_jacobian(differentiation_backend, m, z)
        trace_J = tr(only(J))
        vcat(mz, -trace_J)
    end
    f_aug
end

function augmented_f(
    icnf::CondFFJORD{T, AT},
    mode::TrainMode,
    ys::AbstractVector{<:Real};
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1
    ϵ = convert(AT, randn(rng, T, icnf.nvars))

    function f_aug(u, p, t)
        m = Chain(x -> vcat(x, ys), icnf.re(p))
        z = u[1:(end - 1)]
        v_pb = AbstractDifferentiation.value_and_pullback_function(
            differentiation_backend,
            m,
            z,
        )
        mz, ϵJ = v_pb(ϵ)
        ϵJ = only(ϵJ)
        trace_J = ϵJ ⋅ ϵ
        vcat(mz, -trace_J)
    end
    f_aug
end

@functor CondFFJORD (p,)

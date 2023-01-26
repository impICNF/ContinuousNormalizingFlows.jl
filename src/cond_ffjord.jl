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

function CondFFJORD(
    re::Optimisers.Restructure,
    p::AbstractVector{T},
    nvars::Integer,
    basedist::Distribution,
    tspan::Tuple{T, T},
) where {T <: AbstractFloat, AT <: AbstractArray}
    CondFFJORD{eltype(p), eval(typeof(p).name.name)}(re, p, nvars, basedist, tspan)
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
    n_batch::Integer,
    ys::AbstractMatrix;
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    function f_aug(u, p, t)
        m = Chain(x -> vcat(x, ys), icnf.re(p))
        z = u[1:(end - 1), :]
        mz, J = jacobian_batched(m, z, T, AT)
        trace_J = transpose(tr.(eachslice(J; dims = 3)))
        vcat(mz, -trace_J)
    end
    f_aug
end

function augmented_f(
    icnf::CondFFJORD{T, AT},
    mode::TrainMode,
    n_batch::Integer,
    ys::AbstractMatrix;
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    ϵ = convert(AT, randn(rng, T, icnf.nvars, n_batch))

    function f_aug(u, p, t)
        m = Chain(x -> vcat(x, ys), icnf.re(p))
        z = u[1:(end - 1), :]
        mz, back = Zygote.pullback(m, z)
        ϵJ = only(back(ϵ))
        trace_J = sum(ϵJ .* ϵ; dims = 1)
        vcat(mz, -trace_J)
    end
    f_aug
end

@functor CondFFJORD (p,)

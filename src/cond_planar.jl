export CondPlanar

"""
Implementation of Planar (Conditional Version)
"""
struct CondPlanar{T <: AbstractFloat, AT <: AbstractArray} <: AbstractCondICNF{T, AT}
    re::Optimisers.Restructure
    p::AbstractVector{T}

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    # trace_test
    # trace_train
end

function CondPlanar{T, AT}(
    nn::PlanarNN,
    nvars::Integer,
    ;
    basedist::Distribution = MvNormal(Zeros{T}(nvars), one(T) * I),
    tspan::Tuple{T, T} = convert(Tuple{T, T}, default_tspan),
    rng::AbstractRNG = Random.default_rng(),
) where {T <: AbstractFloat, AT <: AbstractArray}
    nn = fmap(x -> adapt(T, x), nn)
    p, re = destructure(nn)
    CondPlanar{T, AT}(re, convert(AT{T}, p), nvars, basedist, tspan)
end

function augmented_f(
    icnf::CondPlanar{T, AT},
    mode::Mode,
    ys::AbstractVector{<:Real};
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1

    function f_aug(u, p, t)
        m_a = icnf.re(p)
        m = Chain(x -> vcat(x, ys), m_a)
        z = u[1:(end - 1)]
        mz = m(z)
        trace_J = transpose(m_a.u) * only(AbstractDifferentiation.jacobian(differentiation_backend, x -> pl_h(m, x), z))
        vcat(mz, -trace_J)
    end
    f_aug
end

@functor CondPlanar (p,)

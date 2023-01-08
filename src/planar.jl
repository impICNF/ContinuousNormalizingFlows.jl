export Planar, PlanarNN

struct PlanarNN
    u::AbstractVector
    w::AbstractVector
    b::AbstractVector

    h::Function
end

function PlanarNN(
    nvars::Integer,
    h::Function = tanh;
    cond = false,
    rng::AbstractRNG = Random.default_rng(),
)
    u = randn(rng, nvars)
    w = randn(rng, cond ? nvars * 2 : nvars)
    b = randn(rng, 1)
    PlanarNN(u, w, b, h)
end

@functor PlanarNN (u, w, b)

function (m::PlanarNN)(z::AbstractVecOrMat)::AbstractVecOrMat
    u, w, b = m.u, m.w, only(m.b)
    h = NNlib.fast_act(m.h, z)
    u * h.(muladd(transpose(w), z, b))
end

"""
Implementation of Planar Flows from

[Chen, Ricky TQ, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. "Neural Ordinary Differential Equations." arXiv preprint arXiv:1806.07366 (2018).](https://arxiv.org/abs/1806.07366)
"""
struct Planar{T <: AbstractFloat, AT <: AbstractArray} <: AbstractICNF{T, AT}
    re::Optimisers.Restructure
    p::AbstractVector{T}

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    ϵ::AbstractVector{T}

    # trace_test
    # trace_train
end

function Planar(
    re::Optimisers.Restructure,
    p::AbstractVector{T},
    nvars::Integer,
    basedist::Distribution,
    tspan::Tuple{T, T},
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, AT <: AbstractArray}
    Planar{eltype(p), eval(typeof(p).name.name)}(re, p, nvars, basedist, tspan, ϵ)
end

function Planar{T, AT}(
    nn::PlanarNN,
    nvars::Integer,
    ;
    basedist::Distribution = MvNormal(Zeros{T}(nvars), one(T) * I),
    tspan::Tuple{T, T} = convert(Tuple{T, T}, default_tspan),
    rng::AbstractRNG = Random.default_rng(),
) where {T <: AbstractFloat, AT <: AbstractArray}
    nn = fmap(x -> adapt(T, x), nn)
    p, re = destructure(nn)
    Planar{T, AT}(
        re,
        convert(AT{T}, p),
        nvars,
        basedist,
        tspan,
        convert(AT, randn(rng, T, nvars)),
    )
end

function augmented_f(
    icnf::Planar{T, AT},
    mode::TestMode,
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    function f_aug(u, p, t)
        m = icnf.re(p)
        z = u[1:(end - 1), :]
        mz, J = jacobian_batched(m, z, T, AT)
        trace_J = transpose(tr.(eachslice(J; dims = 3)))
        vcat(mz, -trace_J)
    end
    f_aug
end

function augmented_f(
    icnf::Planar{T, AT},
    mode::TrainMode,
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    function f_aug(u, p, t)
        m = icnf.re(p)
        z = u[1:(end - 1), :]
        mz, back = Zygote.pullback(m, z)
        ϵJ = only(back(repeat(icnf.ϵ, 1, size(z, 2))))
        trace_J = transpose(icnf.ϵ) * ϵJ
        vcat(mz, -trace_J)
    end
    f_aug
end

function inference(
    icnf::Planar{T, AT},
    mode::TestMode,
    xs::AbstractMatrix,
    p::AbstractVector = icnf.p,
    args...;
    kwargs...,
)::AbstractVector where {T <: AbstractFloat, AT <: AbstractArray}
    zrs = convert(AT, zeros(T, 1, size(xs, 2)))
    f_aug = augmented_f(icnf, mode)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(xs, zrs), icnf.tspan, p, args...; kwargs...)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:(end - 1), :]
    Δlogp = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x
end

function inference(
    icnf::Planar{T, AT},
    mode::TrainMode,
    xs::AbstractMatrix,
    p::AbstractVector = icnf.p,
    args...;
    kwargs...,
)::AbstractVector where {T <: AbstractFloat, AT <: AbstractArray}
    zrs = convert(AT, zeros(T, 1, size(xs, 2)))
    f_aug = augmented_f(icnf, mode)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(xs, zrs), icnf.tspan, p, args...; kwargs...)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:(end - 1), :]
    Δlogp = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x
end

function generate(
    icnf::Planar{T, AT},
    mode::TestMode,
    n::Integer,
    p::AbstractVector = icnf.p,
    args...;
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)::AbstractMatrix{T} where {T <: AbstractFloat, AT <: AbstractArray}
    new_xs = convert(AT, rand(rng, icnf.basedist, n))
    zrs = convert(AT, zeros(T, 1, size(new_xs, 2)))
    f_aug = augmented_f(icnf, mode)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(new_xs, zrs), reverse(icnf.tspan), p, args...; kwargs...)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:(end - 1), :]
    z
end

function generate(
    icnf::Planar{T, AT},
    mode::TrainMode,
    n::Integer,
    p::AbstractVector = icnf.p,
    args...;
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)::AbstractMatrix{T} where {T <: AbstractFloat, AT <: AbstractArray}
    new_xs = convert(AT, rand(rng, icnf.basedist, n))
    zrs = convert(AT, zeros(T, 1, size(new_xs, 2)))
    f_aug = augmented_f(icnf, mode)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(new_xs, zrs), reverse(icnf.tspan), p, args...; kwargs...)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:(end - 1), :]
    z
end

@functor Planar (p,)

function loss(
    icnf::Planar{T, AT},
    xs::AbstractMatrix,
    p::AbstractVector = icnf.p;
    agg::Function = mean,
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x = inference(icnf, TrainMode(), xs, p)
    agg(-logp̂x)
end

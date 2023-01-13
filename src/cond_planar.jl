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

function CondPlanar(
    re::Optimisers.Restructure,
    p::AbstractVector{T},
    nvars::Integer,
    basedist::Distribution,
    tspan::Tuple{T, T},
) where {T <: AbstractFloat, AT <: AbstractArray}
    CondPlanar{eltype(p), eval(typeof(p).name.name)}(re, p, nvars, basedist, tspan)
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
    CondPlanar{T, AT}(
        re,
        convert(AT{T}, p),
        nvars,
        basedist,
        tspan,
        convert(AT, randn(rng, T, nvars)),
    )
end

function augmented_f(
    icnf::CondPlanar{T, AT},
    mode::TestMode,
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
    icnf::CondPlanar{T, AT},
    mode::TrainMode,
    ys::AbstractMatrix;
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    ϵ = convert(AT, randn(rng, T, icnf.nvars))

    function f_aug(u, p, t)
        m = Chain(x -> vcat(x, ys), icnf.re(p))
        z = u[1:(end - 1), :]
        mz, back = Zygote.pullback(m, z)
        ϵJ = only(back(repeat(ϵ, 1, size(z, 2))))
        trace_J = transpose(ϵ) * ϵJ
        vcat(mz, -trace_J)
    end
    f_aug
end

function inference(
    icnf::CondPlanar{T, AT},
    mode::TestMode,
    xs::AbstractMatrix,
    ys::AbstractMatrix,
    p::AbstractVector = icnf.p,
    args...;
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)::AbstractVector where {T <: AbstractFloat, AT <: AbstractArray}
    zrs = convert(AT, zeros(T, 1, size(xs, 2)))
    f_aug = augmented_f(icnf, mode, ys; rng)
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
    icnf::CondPlanar{T, AT},
    mode::TrainMode,
    xs::AbstractMatrix,
    ys::AbstractMatrix,
    p::AbstractVector = icnf.p,
    args...;
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)::AbstractVector where {T <: AbstractFloat, AT <: AbstractArray}
    zrs = convert(AT, zeros(T, 1, size(xs, 2)))
    f_aug = augmented_f(icnf, mode, ys; rng)
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
    icnf::CondPlanar{T, AT},
    mode::TestMode,
    ys::AbstractMatrix,
    n::Integer,
    p::AbstractVector = icnf.p,
    args...;
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)::AbstractMatrix{T} where {T <: AbstractFloat, AT <: AbstractArray}
    new_xs = convert(AT, rand(rng, icnf.basedist, n))
    zrs = convert(AT, zeros(T, 1, size(new_xs, 2)))
    f_aug = augmented_f(icnf, mode, ys; rng)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(new_xs, zrs), reverse(icnf.tspan), p, args...; kwargs...)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:(end - 1), :]
    z
end

function generate(
    icnf::CondPlanar{T, AT},
    mode::TrainMode,
    ys::AbstractMatrix,
    n::Integer,
    p::AbstractVector = icnf.p,
    args...;
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)::AbstractMatrix{T} where {T <: AbstractFloat, AT <: AbstractArray}
    new_xs = convert(AT, rand(rng, icnf.basedist, n))
    zrs = convert(AT, zeros(T, 1, size(new_xs, 2)))
    f_aug = augmented_f(icnf, mode, ys; rng)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(new_xs, zrs), reverse(icnf.tspan), p, args...; kwargs...)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:(end - 1), :]
    z
end

@functor CondPlanar (p,)

function loss(
    icnf::CondPlanar{T, AT},
    xs::AbstractMatrix,
    ys::AbstractMatrix,
    p::AbstractVector = icnf.p;
    agg::Function = mean,
    rng::AbstractRNG = Random.default_rng(),
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x = inference(icnf, TrainMode(), xs, ys, p; rng)
    agg(-logp̂x)
end

export FFJORD

"""
Implementation of FFJORD from

[Grathwohl, Will, Ricky TQ Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. "Ffjord: Free-form continuous dynamics for scalable reversible generative models." arXiv preprint arXiv:1810.01367 (2018).](https://arxiv.org/abs/1810.01367)
"""
struct FFJORD{T <: AbstractFloat, AT <: AbstractArray} <: AbstractICNF{T, AT}
    re::Optimisers.Restructure
    p::AbstractVector{T}

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    # trace_test
    # trace_train
end

function FFJORD(
    re::Optimisers.Restructure,
    p::AbstractVector{T},
    nvars::Integer,
    basedist::Distribution,
    tspan::Tuple{T, T},
) where {T <: AbstractFloat, AT <: AbstractArray}
    FFJORD{eltype(p), eval(typeof(p).name.name)}(re, p, nvars, basedist, tspan)
end

function FFJORD{T, AT}(
    nn,
    nvars::Integer,
    ;
    basedist::Distribution = MvNormal(Zeros{T}(nvars), one(T) * I),
    tspan::Tuple{T, T} = convert(Tuple{T, T}, default_tspan),
    rng::AbstractRNG = Random.default_rng(),
) where {T <: AbstractFloat, AT <: AbstractArray}
    nn = fmap(x -> adapt(T, x), nn)
    p, re = destructure(nn)
    FFJORD{T, AT}(re, convert(AT{T}, p), nvars, basedist, tspan)
end

function augmented_f(
    icnf::FFJORD{T, AT},
    mode::TestMode,
    n_batch::Integer;
    rng::AbstractRNG = Random.default_rng(),
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
    icnf::FFJORD{T, AT},
    mode::TrainMode,
    n_batch::Integer;
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    ϵ = convert(AT, randn(rng, T, icnf.nvars, n_batch))

    function f_aug(u, p, t)
        m = icnf.re(p)
        z = u[1:(end - 1), :]
        mz, back = Zygote.pullback(m, z)
        ϵJ = only(back(ϵ))
        trace_J = sum(ϵJ .* ϵ, dims=1)
        vcat(mz, -trace_J)
    end
    f_aug
end

function inference(
    icnf::FFJORD{T, AT},
    mode::TestMode,
    xs::AbstractMatrix,
    p::AbstractVector = icnf.p,
    args...;
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)::AbstractVector where {T <: AbstractFloat, AT <: AbstractArray}
    zrs = convert(AT, zeros(T, 1, size(xs, 2)))
    f_aug = augmented_f(icnf, mode, size(xs, 2); rng)
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
    icnf::FFJORD{T, AT},
    mode::TrainMode,
    xs::AbstractMatrix,
    p::AbstractVector = icnf.p,
    args...;
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)::AbstractVector where {T <: AbstractFloat, AT <: AbstractArray}
    zrs = convert(AT, zeros(T, 1, size(xs, 2)))
    f_aug = augmented_f(icnf, mode, size(xs, 2); rng)
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
    icnf::FFJORD{T, AT},
    mode::TestMode,
    n::Integer,
    p::AbstractVector = icnf.p,
    args...;
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)::AbstractMatrix{T} where {T <: AbstractFloat, AT <: AbstractArray}
    new_xs = convert(AT, rand(rng, icnf.basedist, n))
    zrs = convert(AT, zeros(T, 1, size(new_xs, 2)))
    f_aug = augmented_f(icnf, mode, size(new_xs, 2); rng)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(new_xs, zrs), reverse(icnf.tspan), p, args...; kwargs...)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:(end - 1), :]
    z
end

function generate(
    icnf::FFJORD{T, AT},
    mode::TrainMode,
    n::Integer,
    p::AbstractVector = icnf.p,
    args...;
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)::AbstractMatrix{T} where {T <: AbstractFloat, AT <: AbstractArray}
    new_xs = convert(AT, rand(rng, icnf.basedist, n))
    zrs = convert(AT, zeros(T, 1, size(new_xs, 2)))
    f_aug = augmented_f(icnf, mode, size(new_xs, 2); rng)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(new_xs, zrs), reverse(icnf.tspan), p, args...; kwargs...)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:(end - 1), :]
    z
end

@functor FFJORD (p,)

function loss(
    icnf::FFJORD{T, AT},
    xs::AbstractMatrix,
    p::AbstractVector = icnf.p;
    agg::Function = mean,
    rng::AbstractRNG = Random.default_rng(),
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x = inference(icnf, TrainMode(), xs, p; rng)
    agg(-logp̂x)
end

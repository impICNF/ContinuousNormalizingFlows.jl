export FFJORD

"""
Implementation of FFJORD from

[Grathwohl, Will, Ricky TQ Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. "Ffjord: Free-form continuous dynamics for scalable reversible generative models." arXiv preprint arXiv:1810.01367 (2018).](https://arxiv.org/abs/1810.01367)
"""
struct FFJORD{T <: AbstractFloat} <: AbstractICNF{T}
    re::Optimisers.Restructure
    p::AbstractVector{T}

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    solvealg_test::SciMLBase.AbstractODEAlgorithm
    solvealg_train::SciMLBase.AbstractODEAlgorithm

    sensealg_test::SciMLBase.AbstractSensitivityAlgorithm
    sensealg_train::SciMLBase.AbstractSensitivityAlgorithm

    acceleration::AbstractResource

    array_mover::Function
    ϵ::AbstractVector{T}

    # trace_test
    # trace_train
end

function FFJORD{T}(
        nn,
        nvars::Integer,
        ;
        basedist::Distribution=MvNormal(Zeros{T}(nvars), one(T)*I),
        tspan::Tuple{T, T}=convert(Tuple{T, T}, (0, 1)),

        solvealg_test::SciMLBase.AbstractODEAlgorithm=default_solvealg,
        solvealg_train::SciMLBase.AbstractODEAlgorithm=default_solvealg,

        sensealg_test::SciMLBase.AbstractSensitivityAlgorithm=default_sensealg,
        sensealg_train::SciMLBase.AbstractSensitivityAlgorithm=default_sensealg,

        acceleration::AbstractResource=default_acceleration,

        rng::AbstractRNG=Random.default_rng(),
        ) where {T <: AbstractFloat}
    array_mover = make_mover(acceleration, T)
    nn = fmap(x -> adapt(T, x), nn)
    p, re = destructure(nn)
    FFJORD{T}(
        re, p |> array_mover, nvars, basedist, tspan,
        solvealg_test, solvealg_train,
        sensealg_test, sensealg_train,
        acceleration, array_mover, randn(rng, T, nvars) |> array_mover,
    )
end

function augmented_f(icnf::FFJORD{T}, mode::TestMode)::Function where {T <: AbstractFloat}

    function f_aug(u, p, t)
        m = icnf.re(p)
        z = u[1:end - 1, :]
        mz, J = jacobian_batched(m, z, icnf.array_mover)
        trace_J = transpose(tr.(eachslice(J; dims=3)))
        vcat(mz, -trace_J)
    end
    f_aug
end

function augmented_f(icnf::FFJORD{T}, mode::TrainMode)::Function where {T <: AbstractFloat, T2 <: Integer}

    function f_aug(u, p, t)
        m = icnf.re(p)
        z = u[1:end - 1, :]
        mz, back = Zygote.pullback(m, z)
        ϵJ = only(back(icnf.ϵ))
        trace_J = transpose(icnf.ϵ) * ϵJ
        vcat(mz, -trace_J)
    end
    f_aug
end

function inference(icnf::FFJORD{T}, mode::TestMode, xs::AbstractMatrix{T}, p::AbstractVector=icnf.p)::AbstractVector where {T <: AbstractFloat}
    xs = xs |> icnf.array_mover
    zrs = zeros(T, 1, size(xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(xs, zrs), icnf.tspan, p)
    sol = solve(prob, icnf.solvealg_test; sensealg=icnf.sensealg_test)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    Δlogp = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x
end

function inference(icnf::FFJORD{T}, mode::TrainMode, xs::AbstractMatrix{T}, p::AbstractVector=icnf.p)::AbstractVector where {T <: AbstractFloat}
    xs = xs |> icnf.array_mover
    zrs = zeros(T, 1, size(xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(xs, zrs), icnf.tspan, p)
    sol = solve(prob, icnf.solvealg_train; sensealg=icnf.sensealg_train)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    Δlogp = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x
end

function generate(icnf::FFJORD{T}, mode::TestMode, n::Integer, p::AbstractVector=icnf.p; rng::AbstractRNG=Random.default_rng())::AbstractMatrix{T} where {T <: AbstractFloat}
    new_xs = rand(rng, icnf.basedist, n) |> icnf.array_mover
    zrs = zeros(T, 1, size(new_xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(new_xs, zrs), reverse(icnf.tspan), p)
    sol = solve(prob, icnf.solvealg_test; sensealg=icnf.sensealg_test)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    z
end

function generate(icnf::FFJORD{T}, mode::TrainMode, n::Integer, p::AbstractVector=icnf.p; rng::AbstractRNG=Random.default_rng())::AbstractMatrix{T} where {T <: AbstractFloat}
    new_xs = rand(rng, icnf.basedist, n) |> icnf.array_mover
    zrs = zeros(T, 1, size(new_xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(new_xs, zrs), reverse(icnf.tspan), p)
    sol = solve(prob, icnf.solvealg_train; sensealg=icnf.sensealg_train)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    z
end

Flux.@functor FFJORD (p,)

function loss(icnf::FFJORD{T}, xs::AbstractMatrix{T}, p::AbstractVector=icnf.p; agg::Function=mean) where {T <: AbstractFloat}
    logp̂x = inference(icnf, TrainMode(), xs, p)
    agg(-logp̂x)
end

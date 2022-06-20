export Planar, PlanarNN

struct PlanarNN
    u::AbstractVector
    w::AbstractVector
    b::AbstractVector

    h::Function
end

function PlanarNN(nvars::Integer, h::Function=tanh; cond=false, rng::Union{AbstractRNG, Nothing}=nothing)
    u = isnothing(rng) ? randn(nvars) : randn(rng, nvars)
    w = isnothing(rng) ? randn(cond ? nvars*2 : nvars) : randn(rng, cond ? nvars*2 : nvars)
    b = isnothing(rng) ? randn(1) : randn(rng, 1)
    PlanarNN(u, w, b, h)
end

Flux.@functor PlanarNN (u, w, b)

function (m::PlanarNN)(z::AbstractVecOrMat)::AbstractVecOrMat
    u, w, b = m.u, m.w, only(m.b)
    h = NNlib.fast_act(m.h, z)
    u * h.(transpose(w) * z .+ b)
end

"""
Implementation of Planar Flows from

[Chen, Ricky TQ, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. "Neural Ordinary Differential Equations." arXiv preprint arXiv:1806.07366 (2018).](https://arxiv.org/abs/1806.07366)
"""
struct Planar{T <: AbstractFloat} <: AbstractICNF{T}
    re::Optimisers.Restructure
    p::AbstractVector{T}

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    solver_test::SciMLBase.AbstractODEAlgorithm
    solver_train::SciMLBase.AbstractODEAlgorithm

    sensealg_test::SciMLBase.AbstractSensitivityAlgorithm
    sensealg_train::SciMLBase.AbstractSensitivityAlgorithm

    acceleration::AbstractResource
    array_mover::Function

    # trace_test
    # trace_train
end

function Planar{T}(
        nn::PlanarNN,
        nvars::Integer,
        ;
        basedist::Distribution=MvNormal(zeros(T, nvars), Diagonal(ones(T, nvars))),
        tspan::Tuple{T, T}=convert(Tuple{T, T}, (0, 1)),

        solver_test::SciMLBase.AbstractODEAlgorithm=default_solver_test,
        solver_train::SciMLBase.AbstractODEAlgorithm=default_solver_train,

        sensealg_test::SciMLBase.AbstractSensitivityAlgorithm=default_sensealg,
        sensealg_train::SciMLBase.AbstractSensitivityAlgorithm=default_sensealg,

        acceleration::AbstractResource=default_acceleration,
        ) where {T <: AbstractFloat}
    array_mover = make_mover(acceleration, T)
    p, re = destructure(nn)
    Planar{T}(
        re, p |> array_mover, nvars, basedist, tspan,
        solver_test, solver_train,
        sensealg_test, sensealg_train,
        acceleration, array_mover,
    )
end

function augmented_f(icnf::Planar{T}, sz::Tuple{T2, T2})::Function where {T <: AbstractFloat, T2 <: Integer}
    o_ = ones(T, sz) |> icnf.array_mover

    function f_aug(u, p, t)
        m = icnf.re(p)
        z = u[1:end - 1, :]
        mz, back = Zygote.pullback(m, z)
        J = only(back(o_))
        trace_J = transpose(m.u) * J
        vcat(mz, -trace_J)
    end
    f_aug
end

function inference(icnf::Planar{T}, mode::TestMode, xs::AbstractMatrix{T}, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractVector where {T <: AbstractFloat}
    xs = xs |> icnf.array_mover
    zrs = zeros(T, 1, size(xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, size(xs))
    func = ODEFunction{false, true}(f_aug)
    prob = ODEProblem{false}(func, vcat(xs, zrs), reverse(icnf.tspan), p)
    sol = solve(prob, icnf.solver_test; sensealg=icnf.sensealg_test)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    Δlogp = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x
end

function inference(icnf::Planar{T}, mode::TrainMode, xs::AbstractMatrix{T}, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractVector where {T <: AbstractFloat}
    xs = xs |> icnf.array_mover
    zrs = zeros(T, 1, size(xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, size(xs))
    func = ODEFunction{false, true}(f_aug)
    prob = ODEProblem{false}(func, vcat(xs, zrs), reverse(icnf.tspan), p)
    sol = solve(prob, icnf.solver_train; sensealg=icnf.sensealg_train)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    Δlogp = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x
end

function generate(icnf::Planar{T}, mode::TestMode, n::Integer, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat}
    new_xs = isnothing(rng) ? rand(icnf.basedist, n) : rand(rng, icnf.basedist, n)
    new_xs = new_xs |> icnf.array_mover
    zrs = zeros(T, 1, size(new_xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, size(new_xs))
    func = ODEFunction{false, true}(f_aug)
    prob = ODEProblem{false}(func, vcat(new_xs, zrs), icnf.tspan, p)
    sol = solve(prob, icnf.solver_test; sensealg=icnf.sensealg_test)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    z
end

function generate(icnf::Planar{T}, mode::TrainMode, n::Integer, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat}
    new_xs = isnothing(rng) ? rand(icnf.basedist, n) : rand(rng, icnf.basedist, n)
    new_xs = new_xs |> icnf.array_mover
    zrs = zeros(T, 1, size(new_xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, size(new_xs))
    func = ODEFunction{false, true}(f_aug)
    prob = ODEProblem{false}(func, vcat(new_xs, zrs), icnf.tspan, p)
    sol = solve(prob, icnf.solver_train; sensealg=icnf.sensealg_train)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    z
end

Flux.@functor Planar (p,)

function loss(icnf::Planar{T}, xs::AbstractMatrix{T}, p::AbstractVector=icnf.p; agg::Function=mean) where {T <: AbstractFloat}
    logp̂x = inference(icnf, TrainMode(), xs, p)
    agg(-logp̂x)
end

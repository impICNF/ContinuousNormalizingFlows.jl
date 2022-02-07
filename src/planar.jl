export Planar, PlanarNN

struct PlanarNN{T <: AbstractFloat}
    u::AbstractVector{T}
    w::AbstractVector{T}
    b::AbstractVector{T}

    h::Function
end

function PlanarNN{T}(nvars::Integer, h::Function=tanh; cond=false) where {T <: AbstractFloat}
    u = randn(T, nvars)
    w = randn(T, cond ? nvars*2 : nvars)
    b = randn(T, 1)
    PlanarNN{T}(u, w, b, h)
end

Flux.@functor PlanarNN (u, w, b)

function (m::PlanarNN{T})(z::AbstractVecOrMat{T}) where {T <: AbstractFloat}
    u, w, b = m.u, m.w, only(m.b)
    h = NNlib.fast_act(m.h, z)
    u * h.(transpose(w) * z .+ b)
end

"""
Implementations of Planar Flows from

[Chen, Ricky TQ, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. "Neural Ordinary Differential Equations." arXiv preprint arXiv:1806.07366 (2018).](https://arxiv.org/abs/1806.07366)
"""
struct Planar{T <: AbstractFloat} <: AbstractICNF{T}
    re::Function
    p::AbstractVector{T}

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    solver_test::SciMLBase.AbstractODEAlgorithm
    solver_train::SciMLBase.AbstractODEAlgorithm

    sensealg_test::SciMLBase.AbstractSensitivityAlgorithm
    sensealg_train::SciMLBase.AbstractSensitivityAlgorithm

    acceleration::AbstractResource

    # trace_test
    # trace_train
end

function Planar{T}(
        nn::PlanarNN{T},
        nvars::Integer,
        ;
        basedist::Distribution=MvNormal(zeros(T, nvars), Diagonal(ones(T, nvars))),
        tspan::Tuple{T, T}=convert.(T, (0, 1)),

        solver_test::SciMLBase.AbstractODEAlgorithm=default_solver_test,
        solver_train::SciMLBase.AbstractODEAlgorithm=default_solver_train,

        sensealg_test::SciMLBase.AbstractSensitivityAlgorithm=default_sensealg,
        sensealg_train::SciMLBase.AbstractSensitivityAlgorithm=default_sensealg,

        acceleration::AbstractResource=default_acceleration,
        ) where {T <: AbstractFloat}
    move = MLJFlux.Mover(acceleration)
    nn = move(nn)
    p, re = Flux.destructure(nn)
    Planar{T}(
        re, p, nvars, basedist, tspan,
        solver_test, solver_train,
        sensealg_test, sensealg_train,
        acceleration,
    )
end

function augmented_f(icnf::Planar{T}, sz::Tuple{T2, T2})::Function where {T <: AbstractFloat, T2 <: Integer}
    move = MLJFlux.Mover(icnf.acceleration)
    o_ = ones(T, sz) |> move

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

function inference(icnf::Planar{T}, mode::TestMode, xs::AbstractMatrix{T})::AbstractVector{T} where {T <: AbstractFloat}
    move = MLJFlux.Mover(icnf.acceleration)
    xs = xs |> move
    zrs = zeros(T, 1, size(xs, 2)) |> move
    f_aug = augmented_f(icnf, size(xs))
    prob = ODEProblem{false}(f_aug, vcat(xs, zrs), icnf.tspan, icnf.p)
    sol = solve(prob, icnf.solver_test; sensealg=icnf.sensealg_test)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    Δlogp = fsol[end, :]
    logp̂x = convert.(T, logpdf(icnf.basedist, z)) - Δlogp
    logp̂x
end

function inference(icnf::Planar{T}, mode::TrainMode, xs::AbstractMatrix{T})::AbstractVector{T} where {T <: AbstractFloat}
    move = MLJFlux.Mover(icnf.acceleration)
    xs = xs |> move
    zrs = zeros(T, 1, size(xs, 2)) |> move
    f_aug = augmented_f(icnf, size(xs))
    prob = ODEProblem{false}(f_aug, vcat(xs, zrs), icnf.tspan, icnf.p)
    sol = solve(prob, icnf.solver_train; sensealg=icnf.sensealg_train)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    Δlogp = fsol[end, :]
    logp̂x = convert.(T, logpdf(icnf.basedist, z)) - Δlogp
    logp̂x
end

function generate(icnf::Planar{T}, mode::TestMode, n::Integer; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat}
    move = MLJFlux.Mover(icnf.acceleration)
    new_xs = isnothing(rng) ? rand(icnf.basedist, n) : rand(rng, icnf.basedist, n)
    new_xs = new_xs |> move
    zrs = zeros(T, 1, size(new_xs, 2)) |> move
    f_aug = augmented_f(icnf, size(new_xs))
    prob = ODEProblem{false}(f_aug, vcat(new_xs, zrs), reverse(icnf.tspan), icnf.p)
    sol = solve(prob, icnf.solver_test; sensealg=icnf.sensealg_test)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    z
end

function generate(icnf::Planar{T}, mode::TrainMode, n::Integer; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat}
    move = MLJFlux.Mover(icnf.acceleration)
    new_xs = isnothing(rng) ? rand(icnf.basedist, n) : rand(rng, icnf.basedist, n)
    new_xs = new_xs |> move
    zrs = zeros(T, 1, size(new_xs, 2)) |> move
    f_aug = augmented_f(icnf, size(new_xs))
    prob = ODEProblem{false}(f_aug, vcat(new_xs, zrs), reverse(icnf.tspan), icnf.p)
    sol = solve(prob, icnf.solver_train; sensealg=icnf.sensealg_train)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    z
end

Flux.@functor Planar (p,)

function loss_f(icnf::Planar{T}; agg::Function=mean)::Function where {T <: AbstractFloat}
    function f(x::AbstractMatrix{T})::T
        logp̂x = inference(icnf, TrainMode(), x)
        agg(-logp̂x)
    end
    f
end

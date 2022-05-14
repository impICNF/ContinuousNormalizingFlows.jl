export CondFFJORD

"""
Implementation of FFJORD (Conditional Version)
"""
struct CondFFJORD{T <: AbstractFloat} <: AbstractCondICNF{T}
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

    # trace_test
    # trace_train
end

function CondFFJORD{T}(
        nn,
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
    if T <: Float64
        nn = f64(nn)
    elseif T <: Float32
        nn = f32(nn)
    else
        nn = Flux.paramtype(T, nn)
    end
    nn = move(nn)
    p, re = destructure(nn)
    CondFFJORD{T}(
        re, p, nvars, basedist, tspan,
        solver_test, solver_train,
        sensealg_test, sensealg_train,
        acceleration,
    )
end

function augmented_f(icnf::CondFFJORD{T}, mode::TestMode, ys::Union{AbstractMatrix{T}, CuArray})::Function where {T <: AbstractFloat}
    move = MLJFlux.Mover(icnf.acceleration)

    function f_aug(u, p, t)
        m = Chain(
            x -> vcat(x, ys),
            icnf.re(p),
        )
        z = u[1:end - 1, :]
        mz, J = jacobian_batched(m, z, move)
        trace_J = transpose(tr.(eachslice(J, dims=3)))
        vcat(mz, -trace_J)
    end
    f_aug
end

function augmented_f(icnf::CondFFJORD{T}, mode::TrainMode, ys::Union{AbstractMatrix{T}, CuArray}, sz::Tuple{T2, T2}; rng::Union{AbstractRNG, Nothing}=nothing)::Function where {T <: AbstractFloat, T2 <: Integer}
    move = MLJFlux.Mover(icnf.acceleration)
    ϵ = isnothing(rng) ? randn(T, sz) : randn(rng, T, sz)
    ϵ = ϵ |> move

    function f_aug(u, p, t)
        m = Chain(
            x -> vcat(x, ys),
            icnf.re(p),
        )
        z = u[1:end - 1, :]
        mz, back = Zygote.pullback(m, z)
        ϵJ = only(back(ϵ))
        trace_J = sum(ϵJ .* ϵ, dims=1)
        vcat(mz, -trace_J)
    end
    f_aug
end

function inference(icnf::CondFFJORD{T}, mode::TestMode, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractVector where {T <: AbstractFloat}
    move = MLJFlux.Mover(icnf.acceleration)
    xs = xs |> move
    ys = ys |> move
    zrs = zeros(T, 1, size(xs, 2)) |> move
    f_aug = augmented_f(icnf, mode, ys)
    prob = ODEProblem{false}(f_aug, vcat(xs, zrs), icnf.tspan, p)
    sol = solve(prob, icnf.solver_test; sensealg=icnf.sensealg_test)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    Δlogp = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x
end

function inference(icnf::CondFFJORD{T}, mode::TrainMode, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractVector where {T <: AbstractFloat}
    move = MLJFlux.Mover(icnf.acceleration)
    xs = xs |> move
    ys = ys |> move
    zrs = zeros(T, 1, size(xs, 2)) |> move
    f_aug = augmented_f(icnf, mode, ys, size(xs); rng)
    prob = ODEProblem{false}(f_aug, vcat(xs, zrs), icnf.tspan, p)
    sol = solve(prob, icnf.solver_train; sensealg=icnf.sensealg_train)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    Δlogp = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x
end

function generate(icnf::CondFFJORD{T}, mode::TestMode, ys::AbstractMatrix{T}, n::Integer, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat}
    move = MLJFlux.Mover(icnf.acceleration)
    ys = ys |> move
    new_xs = isnothing(rng) ? rand(icnf.basedist, n) : rand(rng, icnf.basedist, n)
    new_xs = new_xs |> move
    zrs = zeros(T, 1, size(new_xs, 2)) |> move
    f_aug = augmented_f(icnf, mode, ys)
    prob = ODEProblem{false}(f_aug, vcat(new_xs, zrs), reverse(icnf.tspan), p)
    sol = solve(prob, icnf.solver_test; sensealg=icnf.sensealg_test)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    z
end

function generate(icnf::CondFFJORD{T}, mode::TrainMode, ys::AbstractMatrix{T}, n::Integer, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat}
    move = MLJFlux.Mover(icnf.acceleration)
    ys = ys |> move
    new_xs = isnothing(rng) ? rand(icnf.basedist, n) : rand(rng, icnf.basedist, n)
    new_xs = new_xs |> move
    zrs = zeros(T, 1, size(new_xs, 2)) |> move
    f_aug = augmented_f(icnf, mode, ys, size(new_xs))
    prob = ODEProblem{false}(f_aug, vcat(new_xs, zrs), reverse(icnf.tspan), p)
    sol = solve(prob, icnf.solver_train; sensealg=icnf.sensealg_train)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    z
end

Flux.@functor CondFFJORD (p,)

function loss(icnf::CondFFJORD{T}, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p; agg::Function=mean, nλ::T=convert(T, 1e-4)) where {T <: AbstractFloat}
    logp̂x = inference(icnf, TrainMode(), xs, ys, p)
    prm_n = norm(p)
    agg(-logp̂x) + nλ*prm_n
end

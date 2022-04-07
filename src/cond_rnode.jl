export CondRNODE

"""
Implementation of RNODE (Conditional Version)
"""
struct CondRNODE{T <: AbstractFloat} <: AbstractCondICNF{T}
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

function CondRNODE{T}(
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
    p, re = Flux.destructure(nn)
    CondRNODE{T}(
        re, p, nvars, basedist, tspan,
        solver_test, solver_train,
        sensealg_test, sensealg_train,
        acceleration,
    )
end

function augmented_f(icnf::CondRNODE{T}, mode::TestMode, ys::Union{AbstractMatrix{T}, CuArray})::Function where {T <: AbstractFloat}
    move = MLJFlux.Mover(icnf.acceleration)

    function f_aug(u, p, t)
        m = Chain(
            x -> vcat(x, ys),
            icnf.re(p),
        )
        z = u[1:end - 1, :]
        ż, J = jacobian_batched(m, z, move)
        l̇ = transpose(tr.(eachslice(J, dims=3)))
        vcat(ż, -l̇)
    end
    f_aug
end

function augmented_f(icnf::CondRNODE{T}, mode::TrainMode, ys::Union{AbstractMatrix{T}, CuArray}, sz::Tuple{T2, T2})::Function where {T <: AbstractFloat, T2 <: Integer}
    move = MLJFlux.Mover(icnf.acceleration)
    ϵ = randn(T, sz) |> move

    function f_aug(u, p, t)
        m = Chain(
            x -> vcat(x, ys),
            icnf.re(p),
        )
        z = u[1:end - 3, :]
        ż, back = Zygote.pullback(m, z)
        ϵJ = only(back(ϵ))
        l̇ = sum(ϵJ .* ϵ, dims=1)
        Ė = transpose(norm.(eachcol(ż)))
        ṅ = transpose(norm.(eachcol(ϵJ)))
        vcat(ż, -l̇, Ė, ṅ)
    end
    f_aug
end

function inference(icnf::CondRNODE{T}, mode::TestMode, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p)::AbstractVector where {T <: AbstractFloat}
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

function inference(icnf::CondRNODE{T}, mode::TrainMode, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p)::Tuple where {T <: AbstractFloat}
    move = MLJFlux.Mover(icnf.acceleration)
    xs = xs |> move
    ys = ys |> move
    zrs = zeros(T, 3, size(xs, 2)) |> move
    f_aug = augmented_f(icnf, mode, ys, size(xs))
    prob = ODEProblem{false}(f_aug, vcat(xs, zrs), icnf.tspan, p)
    sol = solve(prob, icnf.solver_train; sensealg=icnf.sensealg_train)
    fsol = sol[:, :, end]
    z = fsol[1:end - 3, :]
    Δlogp = fsol[end - 2, :]
    Ė = fsol[end - 1, :]
    ṅ = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x, Ė, ṅ
end

function generate(icnf::CondRNODE{T}, mode::TestMode, ys::AbstractMatrix{T}, n::Integer, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat}
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

function generate(icnf::CondRNODE{T}, mode::TrainMode, ys::AbstractMatrix{T}, n::Integer, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat}
    move = MLJFlux.Mover(icnf.acceleration)
    ys = ys |> move
    new_xs = isnothing(rng) ? rand(icnf.basedist, n) : rand(rng, icnf.basedist, n)
    new_xs = new_xs |> move
    zrs = zeros(T, 3, size(new_xs, 2)) |> move
    f_aug = augmented_f(icnf, mode, ys, size(new_xs))
    prob = ODEProblem{false}(f_aug, vcat(new_xs, zrs), reverse(icnf.tspan), p)
    sol = solve(prob, icnf.solver_train; sensealg=icnf.sensealg_train)
    fsol = sol[:, :, end]
    z = fsol[1:end - 3, :]
    z
end

Flux.@functor CondRNODE (p,)

function loss_f(icnf::CondRNODE{T}, opt_app::FluxOptApp, λ₁::T=convert(T, 1e-2), λ₂::T=convert(T, 1e-2); agg::Function=mean)::Function where {T <: AbstractFloat}
    function f(x::AbstractMatrix{T}, y::AbstractMatrix{T})::T
        logp̂x, Ė, ṅ = inference(icnf, TrainMode(), x, y)
        agg(-logp̂x + λ₁*Ė + λ₂*ṅ)
    end
    f
end

function loss_f(icnf::CondRNODE{T}, opt_app::SciMLOptApp, λ₁::T=convert(T, 1e-2), λ₂::T=convert(T, 1e-2); agg::Function=mean)::Function where {T <: AbstractFloat}
    function f(θ::AbstractVector, p::SciMLBase.NullParameters, x::AbstractMatrix{T}, y::AbstractMatrix{T})
        logp̂x, Ė, ṅ = inference(icnf, TrainMode(), x, y, θ)
        agg(-logp̂x + λ₁*Ė + λ₂*ṅ)
    end
    f
end

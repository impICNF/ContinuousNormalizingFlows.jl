export CondRNODE

"""
Implementation of RNODE (Conditional Version)
"""
struct CondRNODE{T <: AbstractFloat} <: AbstractCondICNF{T}
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

function CondRNODE{T}(
        nn,
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
    nn = fmap(x -> adapt(T, x), nn)
    p, re = destructure(nn)
    CondRNODE{T}(
        re, p |> array_mover, nvars, basedist, tspan,
        solver_test, solver_train,
        sensealg_test, sensealg_train,
        acceleration, array_mover,
    )
end

function augmented_f(icnf::CondRNODE{T}, mode::TestMode, ys::Union{AbstractMatrix{T}, CuArray{T, 2}})::Function where {T <: AbstractFloat}

    function f_aug(u, p, t)
        m = Chain(
            x -> vcat(x, ys),
            icnf.re(p),
        )
        z = u[1:end - 1, :]
        ż, J = jacobian_batched(m, z, icnf.array_mover)
        l̇ = transpose(tr.(eachslice(J; dims=3)))
        vcat(ż, -l̇)
    end
    f_aug
end

function augmented_f(icnf::CondRNODE{T}, mode::TrainMode, ys::Union{AbstractMatrix{T}, CuArray{T, 2}}, sz::Tuple{T2, T2}; rng::Union{AbstractRNG, Nothing}=nothing)::Function where {T <: AbstractFloat, T2 <: Integer}
    ϵ = isnothing(rng) ? randn(T, sz) : randn(rng, T, sz)
    ϵ = ϵ |> icnf.array_mover

    function f_aug(u, p, t)
        m = Chain(
            x -> vcat(x, ys),
            icnf.re(p),
        )
        z = u[1:end - 3, :]
        ż, back = Zygote.pullback(m, z)
        ϵJ = only(back(ϵ))
        l̇ = sum(ϵJ .* ϵ; dims=1)
        Ė = transpose(norm.(eachcol(ż)))
        ṅ = transpose(norm.(eachcol(ϵJ)))
        vcat(ż, -l̇, Ė, ṅ)
    end
    f_aug
end

function inference(icnf::CondRNODE{T}, mode::TestMode, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractVector where {T <: AbstractFloat}
    xs = xs |> icnf.array_mover
    ys = ys |> icnf.array_mover
    zrs = zeros(T, 1, size(xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode, ys)
    func = ODEFunction{false, true}(f_aug)
    prob = ODEProblem{false}(func, vcat(xs, zrs), icnf.tspan, p)
    sol = solve(prob, icnf.solver_test; sensealg=icnf.sensealg_test)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    Δlogp = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x
end

function inference(icnf::CondRNODE{T}, mode::TrainMode, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::Tuple where {T <: AbstractFloat}
    xs = xs |> icnf.array_mover
    ys = ys |> icnf.array_mover
    zrs = zeros(T, 3, size(xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode, ys, size(xs); rng)
    func = ODEFunction{false, true}(f_aug)
    prob = ODEProblem{false}(func, vcat(xs, zrs), icnf.tspan, p)
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
    ys = ys |> icnf.array_mover
    new_xs = isnothing(rng) ? rand(icnf.basedist, n) : rand(rng, icnf.basedist, n)
    new_xs = new_xs |> icnf.array_mover
    zrs = zeros(T, 1, size(new_xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode, ys)
    func = ODEFunction{false, true}(f_aug)
    prob = ODEProblem{false}(func, vcat(new_xs, zrs), reverse(icnf.tspan), p)
    sol = solve(prob, icnf.solver_test; sensealg=icnf.sensealg_test)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    z
end

function generate(icnf::CondRNODE{T}, mode::TrainMode, ys::AbstractMatrix{T}, n::Integer, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat}
    ys = ys |> icnf.array_mover
    new_xs = isnothing(rng) ? rand(icnf.basedist, n) : rand(rng, icnf.basedist, n)
    new_xs = new_xs |> icnf.array_mover
    zrs = zeros(T, 3, size(new_xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode, ys, size(new_xs))
    func = ODEFunction{false, true}(f_aug)
    prob = ODEProblem{false}(func, vcat(new_xs, zrs), reverse(icnf.tspan), p)
    sol = solve(prob, icnf.solver_train; sensealg=icnf.sensealg_train)
    fsol = sol[:, :, end]
    z = fsol[1:end - 3, :]
    z
end

Flux.@functor CondRNODE (p,)

function loss(icnf::CondRNODE{T}, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p, λ₁::T=convert(T, 1e-2), λ₂::T=convert(T, 1e-2); agg::Function=mean) where {T <: AbstractFloat}
    logp̂x, Ė, ṅ = inference(icnf, TrainMode(), xs, ys, p)
    agg(-logp̂x + λ₁*Ė + λ₂*ṅ)
end

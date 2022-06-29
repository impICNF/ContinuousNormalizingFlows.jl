export CondPlanar

"""
Implementation of Planar (Conditional Version)
"""
struct CondPlanar{T <: AbstractFloat} <: AbstractCondICNF{T}
    re::Optimisers.Restructure
    p::AbstractVector{T}

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    differentiation_backend_test::AbstractDifferentiation.AbstractBackend
    differentiation_backend_train::AbstractDifferentiation.AbstractBackend

    solver_test::SciMLBase.AbstractODEAlgorithm
    solver_train::SciMLBase.AbstractODEAlgorithm

    sensealg_test::SciMLBase.AbstractSensitivityAlgorithm
    sensealg_train::SciMLBase.AbstractSensitivityAlgorithm

    acceleration::AbstractResource
    array_mover::Function

    # trace_test
    # trace_train
end

function CondPlanar{T}(
        nn::PlanarNN,
        nvars::Integer,
        ;
        basedist::Distribution=MvNormal(zeros(T, nvars), Diagonal(ones(T, nvars))),
        tspan::Tuple{T, T}=convert(Tuple{T, T}, (0, 1)),

        differentiation_backend_test::AbstractDifferentiation.AbstractBackend=default_differentiation_backend(),
        differentiation_backend_train::AbstractDifferentiation.AbstractBackend=default_differentiation_backend(),

        solver_test::SciMLBase.AbstractODEAlgorithm=default_solver,
        solver_train::SciMLBase.AbstractODEAlgorithm=default_solver,

        sensealg_test::SciMLBase.AbstractSensitivityAlgorithm=default_sensealg,
        sensealg_train::SciMLBase.AbstractSensitivityAlgorithm=default_sensealg,

        acceleration::AbstractResource=default_acceleration,
        ) where {T <: AbstractFloat}
    array_mover = make_mover(acceleration, T)
    nn = fmap(x -> adapt(T, x), nn)
    p, re = destructure(nn)
    CondPlanar{T}(
        re, p |> array_mover, nvars, basedist, tspan,
        differentiation_backend_test, differentiation_backend_train,
        solver_test, solver_train,
        sensealg_test, sensealg_train,
        acceleration, array_mover,
    )
end

function augmented_f(icnf::CondPlanar{T}, mode::TestMode, ys::AbstractMatrix{T})::Function where {T <: AbstractFloat, T2 <: Integer}

    function f_aug(u, p, t)
        m_ = icnf.re(p)
        m = Chain(
            x -> vcat(x, ys),
            m_,
        )
        z = u[1:end - 1, :]
        mz, J = AbstractDifferentiation.value_and_jacobian(icnf.differentiation_backend_train, m, z)
        J = only(J)
        trace_J = transpose(m_.u) * J
        vcat(mz, -trace_J)
    end
    f_aug
end

function augmented_f(icnf::CondPlanar{T}, mode::TrainMode, ys::AbstractMatrix{T})::Function where {T <: AbstractFloat, T2 <: Integer}

    function f_aug(u, p, t)
        m_ = icnf.re(p)
        m = Chain(
            x -> vcat(x, ys),
            m_,
        )
        z = u[1:end - 1, :]
        mz, J = AbstractDifferentiation.value_and_jacobian(icnf.differentiation_backend_train, m, z)
        J = only(J)
        trace_J = transpose(m_.u) * J
        vcat(mz, -trace_J)
    end
    f_aug
end

function inference(icnf::CondPlanar{T}, mode::TestMode, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p; rng::AbstractRNG=Random.default_rng())::AbstractVector where {T <: AbstractFloat}
    xs = xs |> icnf.array_mover
    ys = ys |> icnf.array_mover
    zrs = zeros(T, 1, size(xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode, ys)
    func = ODEFunction{false, true}(f_aug)
    prob = ODEProblem{false, true}(func, vcat(xs, zrs), icnf.tspan, p; alg=icnf.solver_test, sensealg=icnf.sensealg_test)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    Δlogp = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x
end

function inference(icnf::CondPlanar{T}, mode::TrainMode, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p; rng::AbstractRNG=Random.default_rng())::AbstractVector where {T <: AbstractFloat}
    xs = xs |> icnf.array_mover
    ys = ys |> icnf.array_mover
    zrs = zeros(T, 1, size(xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode, ys)
    func = ODEFunction{false, true}(f_aug)
    prob = ODEProblem{false, true}(func, vcat(xs, zrs), icnf.tspan, p; alg=icnf.solver_train, sensealg=icnf.sensealg_train)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    Δlogp = fsol[end, :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x
end

function generate(icnf::CondPlanar{T}, mode::TestMode, ys::AbstractMatrix{T}, n::Integer, p::AbstractVector=icnf.p; rng::AbstractRNG=Random.default_rng())::AbstractMatrix{T} where {T <: AbstractFloat}
    ys = ys |> icnf.array_mover
    new_xs = rand(rng, icnf.basedist, n) |> icnf.array_mover
    zrs = zeros(T, 1, size(new_xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode, ys)
    func = ODEFunction{false, true}(f_aug)
    prob = ODEProblem{false, true}(func, vcat(new_xs, zrs), reverse(icnf.tspan), p; alg=icnf.solver_test, sensealg=icnf.sensealg_test)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    z
end

function generate(icnf::CondPlanar{T}, mode::TrainMode, ys::AbstractMatrix{T}, n::Integer, p::AbstractVector=icnf.p; rng::AbstractRNG=Random.default_rng())::AbstractMatrix{T} where {T <: AbstractFloat}
    ys = ys |> icnf.array_mover
    new_xs = rand(rng, icnf.basedist, n) |> icnf.array_mover
    zrs = zeros(T, 1, size(new_xs, 2)) |> icnf.array_mover
    f_aug = augmented_f(icnf, mode, ys)
    func = ODEFunction{false, true}(f_aug)
    prob = ODEProblem{false, true}(func, vcat(new_xs, zrs), reverse(icnf.tspan), p; alg=icnf.solver_train, sensealg=icnf.sensealg_train)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:end - 1, :]
    z
end

Flux.@functor CondPlanar (p,)

function loss(icnf::CondPlanar{T}, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p; agg::Function=mean) where {T <: AbstractFloat}
    logp̂x = inference(icnf, TrainMode(), xs, ys, p)
    agg(-logp̂x)
end

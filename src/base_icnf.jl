export inference, generate, loss

function inference(
    icnf::AbstractICNF{T, AT, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::Any;
    tspan::NTuple{2, T} = icnf.tspan,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)::Tuple{Vararg{Real}} where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode)
    zrs::AT = zeros(T, n_aug + 1)
    f_aug = augmented_f(icnf, mode, st; differentiation_backend, rng)
    func = ODEFunction{false, SciMLBase.FullSpecialize}(f_aug)
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(func, vcat(xs, zrs), tspan, ps)
    sol = solve(prob, sol_args...; sol_kwargs...)
    fsol = @view sol[:, end]
    z = @view fsol[begin:(end - n_aug - 1)]
    Δlogp = fsol[(end - n_aug)]
    augs = @view fsol[(end - n_aug + 1):end]
    logp̂x = logpdf(basedist, z) - Δlogp
    iszero(n_aug) ? (logp̂x,) : (logp̂x, augs...)
end

function inference(
    icnf::AbstractICNF{T, AT, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::Any;
    tspan::NTuple{2, T} = icnf.tspan,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)::Tuple{Vararg{AbstractVector{<:Real}}} where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode)
    zrs::AT = zeros(T, n_aug + 1, size(xs, 2))
    f_aug = augmented_f(icnf, mode, st, size(xs, 2); differentiation_backend, rng)
    func = ODEFunction{false, SciMLBase.FullSpecialize}(f_aug)
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(func, vcat(xs, zrs), tspan, ps)
    sol = solve(prob, sol_args...; sol_kwargs...)
    fsol = @view sol[:, :, end]
    z = @view fsol[begin:(end - n_aug - 1), :]
    Δlogp = @view fsol[(end - n_aug), :]
    augs = @view fsol[(end - n_aug + 1):end, :]
    logp̂x = logpdf(basedist, z) - Δlogp
    iszero(n_aug) ? (logp̂x,) : (logp̂x, eachrow(augs)...)
end

function generate(
    icnf::AbstractICNF{T, AT, <:VectorMode},
    mode::Mode,
    ps::Any,
    st::Any;
    tspan::NTuple{2, T} = icnf.tspan,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)::AbstractVector{<:Real} where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode)
    new_xs::AT = rand(rng, basedist)
    zrs::AT = zeros(T, n_aug + 1)
    f_aug = augmented_f(icnf, mode, st; differentiation_backend, rng)
    func = ODEFunction{false, SciMLBase.FullSpecialize}(f_aug)
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        vcat(new_xs, zrs),
        reverse(tspan),
        ps,
    )
    sol = solve(prob, sol_args...; sol_kwargs...)
    fsol = @view sol[:, end]
    z = @view fsol[begin:(end - n_aug - 1)]
    z
end

function generate(
    icnf::AbstractICNF{T, AT, <:MatrixMode},
    mode::Mode,
    ps::Any,
    st::Any,
    n::Integer;
    tspan::NTuple{2, T} = icnf.tspan,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)::AbstractMatrix{<:Real} where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode)
    new_xs::AT = rand(rng, basedist, n)
    zrs::AT = zeros(T, n_aug + 1, size(new_xs, 2))
    f_aug = augmented_f(icnf, mode, st, size(new_xs, 2); differentiation_backend, rng)
    func = ODEFunction{false, SciMLBase.FullSpecialize}(f_aug)
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        vcat(new_xs, zrs),
        reverse(tspan),
        ps,
    )
    sol = solve(prob, sol_args...; sol_kwargs...)
    fsol = @view sol[:, :, end]
    z = @view fsol[begin:(end - n_aug - 1), :]
    z
end

function loss(
    icnf::AbstractICNF{T, AT, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::Any;
    tspan::NTuple{2, T} = icnf.tspan,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x, = inference(
        icnf,
        mode,
        xs,
        ps,
        st;
        tspan,
        basedist,
        differentiation_backend,
        rng,
        sol_args,
        sol_kwargs,
    )
    -logp̂x
end

function loss(
    icnf::AbstractICNF{T, AT, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::Any;
    tspan::NTuple{2, T} = icnf.tspan,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x, = inference(
        icnf,
        mode,
        xs,
        ps,
        st;
        tspan,
        basedist,
        differentiation_backend,
        rng,
        sol_args,
        sol_kwargs,
    )
    mean(-logp̂x)
end

export inference, generate, loss

function inference_prob(
    icnf::AbstractICNF{T, AT, CM},
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
)::ODEProblem where {T <: AbstractFloat, AT <: AbstractArray, CM <: VectorMode}
    n_aug = n_augment(icnf, mode)
    zrs = zeros_T_AT(icnf, n_aug + 1)
    f_aug = augmented_f(icnf, mode, st; differentiation_backend, rng)
    func = ODEFunction{false, SciMLBase.FullSpecialize}(f_aug)
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(func, vcat(xs, zrs), tspan, ps)
    prob
end

function inference(
    icnf::AbstractICNF{T, AT, CM},
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
)::Tuple{Vararg{Real}} where {T <: AbstractFloat, AT <: AbstractArray, CM <: VectorMode}
    prob = inference_prob(
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
    n_aug = n_augment(icnf, mode)
    sol = solve(prob, sol_args...; sol_kwargs...)
    fsol = @view sol[:, end]
    z = @view fsol[begin:(end - n_aug - 1)]
    Δlogp = fsol[(end - n_aug)]
    augs = @view fsol[(end - n_aug + 1):end]
    logp̂x = logpdf(basedist, z) - Δlogp
    iszero(n_aug) ? (logp̂x,) : (logp̂x, augs...)
end

function inference_prob(
    icnf::AbstractICNF{T, AT, CM},
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
)::ODEProblem where {T <: AbstractFloat, AT <: AbstractArray, CM <: MatrixMode}
    n_aug = n_augment(icnf, mode)
    zrs = zeros_T_AT(icnf, n_aug + 1, size(xs, 2))
    f_aug = augmented_f(icnf, mode, st, size(xs, 2); differentiation_backend, rng)
    func = ODEFunction{false, SciMLBase.FullSpecialize}(f_aug)
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(func, vcat(xs, zrs), tspan, ps)
    prob
end

function inference(
    icnf::AbstractICNF{T, AT, CM},
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
)::Tuple{
    Vararg{AbstractVector{<:Real}},
} where {T <: AbstractFloat, AT <: AbstractArray, CM <: MatrixMode}
    prob = inference_prob(
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
    n_aug = n_augment(icnf, mode)
    sol = solve(prob, sol_args...; sol_kwargs...)
    fsol = @view sol[:, :, end]
    z = @view fsol[begin:(end - n_aug - 1), :]
    Δlogp = @view fsol[(end - n_aug), :]
    augs = @view fsol[(end - n_aug + 1):end, :]
    logp̂x = logpdf(basedist, z) - Δlogp
    iszero(n_aug) ? (logp̂x,) : (logp̂x, eachrow(augs)...)
end

function generate_prob(
    icnf::AbstractICNF{T, AT, CM},
    mode::Mode,
    ps::Any,
    st::Any;
    tspan::NTuple{2, T} = icnf.tspan,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)::ODEProblem where {T <: AbstractFloat, AT <: AbstractArray, CM <: VectorMode}
    n_aug = n_augment(icnf, mode)
    new_xs::AT = rand(rng, basedist)
    zrs = zeros_T_AT(icnf, n_aug + 1)
    f_aug = augmented_f(icnf, mode, st; differentiation_backend, rng)
    func = ODEFunction{false, SciMLBase.FullSpecialize}(f_aug)
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        vcat(new_xs, zrs),
        reverse(tspan),
        ps,
    )
    prob
end
function generate(
    icnf::AbstractICNF{T, AT, CM},
    mode::Mode,
    ps::Any,
    st::Any;
    tspan::NTuple{2, T} = icnf.tspan,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)::AbstractVector{<:Real} where {T <: AbstractFloat, AT <: AbstractArray, CM <: VectorMode}
    prob = generate_prob(
        icnf,
        mode,
        ps,
        st;
        tspan,
        basedist,
        differentiation_backend,
        rng,
        sol_args,
        sol_kwargs,
    )
    n_aug = n_augment(icnf, mode)
    sol = solve(prob, sol_args...; sol_kwargs...)
    fsol = @view sol[:, end]
    z = @view fsol[begin:(end - n_aug - 1)]
    z
end

function generate_prob(
    icnf::AbstractICNF{T, AT, CM},
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
)::ODEProblem where {T <: AbstractFloat, AT <: AbstractArray, CM <: MatrixMode}
    n_aug = n_augment(icnf, mode)
    new_xs::AT = rand(rng, basedist, n)
    zrs = zeros_T_AT(icnf, n_aug + 1, size(new_xs, 2))
    f_aug = augmented_f(icnf, mode, st, size(new_xs, 2); differentiation_backend, rng)
    func = ODEFunction{false, SciMLBase.FullSpecialize}(f_aug)
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        vcat(new_xs, zrs),
        reverse(tspan),
        ps,
    )
    prob
end

function generate(
    icnf::AbstractICNF{T, AT, CM},
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
)::AbstractMatrix{<:Real} where {T <: AbstractFloat, AT <: AbstractArray, CM <: MatrixMode}
    prob = generate_prob(
        icnf,
        mode,
        ps,
        st,
        n;
        tspan,
        basedist,
        differentiation_backend,
        rng,
        sol_args,
        sol_kwargs,
    )
    n_aug = n_augment(icnf, mode)
    sol = solve(prob, sol_args...; sol_kwargs...)
    fsol = @view sol[:, :, end]
    z = @view fsol[begin:(end - n_aug - 1), :]
    z
end

function loss(
    icnf::AbstractICNF{T, AT, CM},
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
)::Real where {T <: AbstractFloat, AT <: AbstractArray, CM <: VectorMode}
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
    icnf::AbstractICNF{T, AT, CM},
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
)::Real where {T <: AbstractFloat, AT <: AbstractArray, CM <: MatrixMode}
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

function augmented_f(
    icnf::AbstractICNF{T, AT, CM},
    mode::TestMode,
    st::Any;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray, CM <: ADVectorMode}
    n_aug = n_augment(icnf, mode) + 1

    function f_aug(u, p, t)
        z = @view u[begin:(end - n_aug)]
        mz, J = AbstractDifferentiation.value_and_jacobian(
            differentiation_backend,
            x -> first(LuxCore.apply(icnf.nn, x, p, st)),
            z,
        )
        trace_J = tr(only(J))
        vcat(mz, -trace_J)
    end
    f_aug
end

function augmented_f(
    icnf::AbstractICNF{T, AT, CM},
    mode::TestMode,
    st::Any,
    n_batch::Integer;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray, CM <: MatrixMode}
    n_aug = n_augment(icnf, mode) + 1

    function f_aug(u, p, t)
        z = @view u[begin:(end - n_aug), :]
        mz, J = jacobian_batched(icnf, x -> first(LuxCore.apply(icnf.nn, x, p, st)), z)
        trace_J = transpose(tr.(eachslice(J; dims = 3)))
        vcat(mz, -trace_J)
    end
    f_aug
end

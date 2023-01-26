export inference, generate, loss

function inference(
    icnf::AbstractICNF{T, AT},
    mode::Mode,
    xs::AbstractMatrix,
    p::AbstractVector = icnf.p,
    args...;
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)::Tuple where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode)
    zrs = convert(AT, zeros(T, n_aug + 1, size(xs, 2)))
    f_aug = augmented_f(icnf, mode, size(xs, 2); rng)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(xs, zrs), icnf.tspan, p, args...; kwargs...)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:(end - n_aug - 1), :]
    Δlogp = fsol[(end - n_aug), :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    logp̂x, eachrow(fsol[(end - n_aug + 1):end, :])...
end

function generate(
    icnf::AbstractICNF{T, AT},
    mode::Mode,
    n::Integer,
    p::AbstractVector = icnf.p,
    args...;
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)::AbstractMatrix where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode)
    new_xs = convert(AT, rand(rng, icnf.basedist, n))
    zrs = convert(AT, zeros(T, n_aug + 1, size(new_xs, 2)))
    f_aug = augmented_f(icnf, mode, size(new_xs, 2); rng)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(new_xs, zrs), reverse(icnf.tspan), p, args...; kwargs...)
    sol = solve(prob)
    fsol = sol[:, :, end]
    z = fsol[1:(end - n_aug - 1), :]
    z
end

function loss(
    icnf::AbstractICNF{T, AT},
    xs::AbstractMatrix,
    p::AbstractVector = icnf.p;
    agg::Function = mean,
    rng::AbstractRNG = Random.default_rng(),
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x, = inference(icnf, TrainMode(), xs, p; rng)
    agg(-logp̂x)
end

function n_augment(
    icnf::AbstractICNF{T, AT},
    mode::Mode,
)::Integer where {T <: AbstractFloat, AT <: AbstractArray}
    0
end

# pretty-printing
function Base.show(io::IO, icnf::AbstractICNF)
    print(
        io,
        typeof(icnf),
        "\n\tNumber of Variables: ",
        icnf.nvars,
        "\n\tTime Span: ",
        icnf.tspan,
    )
end

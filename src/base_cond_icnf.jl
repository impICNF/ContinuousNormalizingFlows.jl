export inference, generate, loss

function inference(
    icnf::AbstractCondICNF{T, AT},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any,
    args...;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)::Tuple{Vararg{Real}} where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode)
    zrs = convert(AT, zeros(T, n_aug + 1))
    f_aug = augmented_f(icnf, mode, ys, st; differentiation_backend, rng)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(xs, zrs), icnf.tspan, ps)
    sol = solve(prob, args...; kwargs...)
    fsol = sol[:, end]
    z = fsol[1:(end - n_aug - 1)]
    Δlogp = fsol[(end - n_aug)]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    iszero(n_aug) ? (logp̂x,) : (logp̂x, fsol[(end - n_aug + 1):end]...)
end

function inference(
    icnf::AbstractCondICNF{T, AT},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any,
    args...;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)::Base.Iterators.Zip{
    <:Tuple{Vararg{Tuple{Vararg{Real}}}},
} where {T <: AbstractFloat, AT <: AbstractArray}
    zip(
        Folds.map(
            x -> inference(
                icnf,
                mode,
                x,
                ys,
                ps,
                st,
                args...;
                differentiation_backend,
                rng,
                kwargs...,
            ),
            eachcol(xs),
        )...,
    )
end

function generate(
    icnf::AbstractCondICNF{T, AT},
    mode::Mode,
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any,
    args...;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)::AbstractVector{<:Real} where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode)
    new_xs = convert(AT, rand(rng, icnf.basedist))
    zrs = convert(AT, zeros(T, n_aug + 1))
    f_aug = augmented_f(icnf, mode, ys, st; differentiation_backend, rng)
    func = ODEFunction(f_aug)
    prob = ODEProblem(func, vcat(new_xs, zrs), reverse(icnf.tspan), ps)
    sol = solve(prob, args...; kwargs...)
    fsol = sol[:, end]
    z = fsol[1:(end - n_aug - 1)]
    z
end

function generate(
    icnf::AbstractCondICNF{T, AT},
    mode::Mode,
    ys::AbstractVector{<:Real},
    n::Integer,
    ps::Any,
    st::Any,
    args...;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)::AbstractVector{<:AbstractVector{<:Real}} where {T <: AbstractFloat, AT <: AbstractArray}
    Folds.map(
        x -> generate(icnf, mode, ys, ps, st, args...; differentiation_backend, rng, kwargs...),
        1:n,
    )
end

function loss(
    icnf::AbstractCondICNF{T, AT},
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any;
    rng::AbstractRNG = Random.default_rng(),
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x, = inference(icnf, TrainMode(), xs, ys, ps, st; rng)
    -logp̂x
end

function loss(
    icnf::AbstractCondICNF{T, AT},
    xs::AbstractMatrix{<:Real},
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::Any;
    rng::AbstractRNG = Random.default_rng(),
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    Folds.sum(((x, y),) -> loss(icnf, x, y, ps, st; rng), zip(eachcol(xs), eachcol(ys))) /
    size(xs, 2)
end

function n_augment(
    icnf::AbstractCondICNF{T, AT},
    mode::Mode,
)::Integer where {T <: AbstractFloat, AT <: AbstractArray}
    0
end

# pretty-printing
function Base.show(io::IO, icnf::AbstractCondICNF)
    print(
        io,
        typeof(icnf),
        "\n\tNumber of Variables: ",
        icnf.nvars,
        "\n\tTime Span: ",
        icnf.tspan,
    )
end

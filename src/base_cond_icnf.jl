export inference, generate, loss

@views function inference_prob(
    icnf::AbstractCondICNF{T, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{T},
    ys::AbstractVector{T},
    ps::Any,
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    zrs = zeros_T_AT(icnf.resource, icnf, n_aug_input + n_aug + 1)
    ϵ = randn_T_AT(icnf.resource, icnf, icnf.nvars + n_aug_input)
    func = ODEFunction{false, SciMLBase.FullSpecialize}(
        (u, p, t) -> augmented_f(u, p, t, icnf, mode, ys, ϵ, st),
    )
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        cat(xs, zrs; dims = 1),
        steer_tspan(icnf, mode),
        ps;
        icnf.sol_kwargs...,
    )
    prob
end

@views function inference(
    icnf::AbstractCondICNF{T, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{T},
    ys::AbstractVector{T},
    ps::Any,
    st::Any,
) where {T <: AbstractFloat}
    prob = inference_prob(icnf, mode, xs, ys, ps, st)
    n_aug = n_augment(icnf, mode)
    sol = solve(prob, icnf.sol_args...; icnf.sol_kwargs...)
    fsol = sol[:, end]
    z = fsol[begin:(end - n_aug - 1)]
    Δlogp = fsol[(end - n_aug)]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    if iszero(n_aug)
        (logp̂x,)
    else
        augs = fsol[(end - n_aug + 1):end]
        (logp̂x, augs...)
    end
end

@views function inference_prob(
    icnf::AbstractCondICNF{T, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{T},
    ys::AbstractMatrix{T},
    ps::Any,
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    zrs = zeros_T_AT(icnf.resource, icnf, n_aug_input + n_aug + 1, size(xs, 2))
    ϵ = randn_T_AT(icnf.resource, icnf, icnf.nvars + n_aug_input, size(xs, 2))
    func = ODEFunction{false, SciMLBase.FullSpecialize}(
        (u, p, t) -> augmented_f(u, p, t, icnf, mode, ys, ϵ, st),
    )
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        cat(xs, zrs; dims = 1),
        steer_tspan(icnf, mode),
        ps;
        icnf.sol_kwargs...,
    )
    prob
end

@views function inference(
    icnf::AbstractCondICNF{T, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{T},
    ys::AbstractMatrix{T},
    ps::Any,
    st::Any,
) where {T <: AbstractFloat}
    prob = inference_prob(icnf, mode, xs, ys, ps, st)
    n_aug = n_augment(icnf, mode)
    sol = solve(prob, icnf.sol_args...; icnf.sol_kwargs...)
    fsol = sol[:, :, end]
    z = fsol[begin:(end - n_aug - 1), :]
    Δlogp = fsol[(end - n_aug), :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    if iszero(n_aug)
        (logp̂x,)
    else
        augs = fsol[(end - n_aug + 1):end, :]
        (logp̂x, eachrow(augs)...)
    end
end

@views function generate_prob(
    icnf::AbstractCondICNF{T, <:VectorMode},
    mode::Mode,
    ys::AbstractVector{T},
    ps::Any,
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    new_xs = rand_cstm_AT(icnf.resource, icnf, icnf.basedist)
    zrs = zeros_T_AT(icnf.resource, icnf, n_aug + 1)
    ϵ = randn_T_AT(icnf.resource, icnf, icnf.nvars + n_aug_input)
    func = ODEFunction{false, SciMLBase.FullSpecialize}(
        (u, p, t) -> augmented_f(u, p, t, icnf, mode, ys, ϵ, st),
    )
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        cat(new_xs, zrs; dims = 1),
        reverse(steer_tspan(icnf, mode)),
        ps;
        icnf.sol_kwargs...,
    )
    prob
end

@views function generate(
    icnf::AbstractCondICNF{T, <:VectorMode},
    mode::Mode,
    ys::AbstractVector{T},
    ps::Any,
    st::Any,
) where {T <: AbstractFloat}
    prob = generate_prob(icnf, mode, ys, ps, st)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    sol = solve(prob, icnf.sol_args...; icnf.sol_kwargs...)
    fsol = sol[:, end]
    z = fsol[begin:(end - n_aug_input - n_aug - 1)]
    z
end

@views function generate_prob(
    icnf::AbstractCondICNF{T, <:MatrixMode},
    mode::Mode,
    ys::AbstractMatrix{T},
    ps::Any,
    st::Any,
    n::Int,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    new_xs = rand_cstm_AT(icnf.resource, icnf, icnf.basedist, n)
    zrs = zeros_T_AT(icnf.resource, icnf, n_aug + 1, size(new_xs, 2))
    ϵ = randn_T_AT(icnf.resource, icnf, icnf.nvars + n_aug_input, size(new_xs, 2))
    func = ODEFunction{false, SciMLBase.FullSpecialize}(
        (u, p, t) -> augmented_f(u, p, t, icnf, mode, ys, ϵ, st),
    )
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        cat(new_xs, zrs; dims = 1),
        reverse(steer_tspan(icnf, mode)),
        ps;
        icnf.sol_kwargs...,
    )
    prob
end

@views function generate(
    icnf::AbstractCondICNF{T, <:MatrixMode},
    mode::Mode,
    ys::AbstractMatrix{T},
    ps::Any,
    st::Any,
    n::Int,
) where {T <: AbstractFloat}
    prob = generate_prob(icnf, mode, ys, ps, st, n)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    sol = solve(prob, icnf.sol_args...; icnf.sol_kwargs...)
    fsol = sol[:, :, end]
    z = fsol[begin:(end - n_aug_input - n_aug - 1), :]
    z
end

@inline function loss(
    icnf::AbstractCondICNF{T, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{T},
    ys::AbstractVector{T},
    ps::Any,
    st::Any,
) where {T <: AbstractFloat}
    -first(inference(icnf, mode, xs, ys, ps, st))
end

@inline function loss(
    icnf::AbstractCondICNF{T, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{T},
    ys::AbstractMatrix{T},
    ps::Any,
    st::Any,
) where {T <: AbstractFloat}
    -mean(first(inference(icnf, mode, xs, ys, ps, st)))
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:ADVectorMode},
    mode::TestMode,
    ys::AbstractVector{T},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    mz, J = AbstractDifferentiation.value_and_jacobian(
        icnf.differentiation_backend,
        x -> icnf._fnn(cat(x, ys; dims = 1), p, st),
        z,
    )
    trace_J = tr(only(J))
    cat(mz, -trace_J; dims = 1)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:MatrixMode},
    mode::TestMode,
    ys::AbstractMatrix{T},
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    mz, J = jacobian_batched(icnf, x -> icnf._fnn(cat(x, ys; dims = 1), p, st), z)
    trace_J = transpose(tr.(eachslice(J; dims = 3)))
    cat(mz, -trace_J; dims = 1)
end

@inline function (icnf::AbstractCondICNF)(xs_ys::Any, ps::Any, st::Any)
    xs, ys = xs_ys
    first(inference(icnf, TrainMode(), xs, ys, ps, st))
end

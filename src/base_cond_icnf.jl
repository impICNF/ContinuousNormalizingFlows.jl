export inference, generate, loss

@views function inference_prob(
    icnf::AbstractCondICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any,
)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    zrs = zeros_T_AT(icnf.resource, icnf, n_aug_input + n_aug + 1)
    ϵ = randn_T_AT(icnf.resource, icnf, icnf.nvars + n_aug_input)
    ODEProblem{false, SciMLBase.FullSpecialize}(
        let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
            (u, p, t) -> augmented_f(u, p, t, icnf, mode, ys, ϵ, st)
        end,
        cat(xs, zrs; dims = 1),
        steer_tspan(icnf, mode),
        ps;
        icnf.sol_kwargs...,
    )
end

@views function inference_prob(
    icnf::AbstractCondICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::Any,
)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    zrs = zeros_T_AT(icnf.resource, icnf, n_aug_input + n_aug + 1, size(xs, 2))
    ϵ = randn_T_AT(icnf.resource, icnf, icnf.nvars + n_aug_input, size(xs, 2))
    ODEProblem{false, SciMLBase.FullSpecialize}(
        let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
            (u, p, t) -> augmented_f(u, p, t, icnf, mode, ys, ϵ, st)
        end,
        cat(xs, zrs; dims = 1),
        steer_tspan(icnf, mode),
        ps;
        icnf.sol_kwargs...,
    )
end

@views function generate_prob(
    icnf::AbstractCondICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any,
)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    new_xs = rand_cstm_AT(icnf.resource, icnf, icnf.basedist)
    zrs = zeros_T_AT(icnf.resource, icnf, n_aug + 1)
    ϵ = randn_T_AT(icnf.resource, icnf, icnf.nvars + n_aug_input)
    ODEProblem{false, SciMLBase.FullSpecialize}(
        let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
            (u, p, t) -> augmented_f(u, p, t, icnf, mode, ys, ϵ, st)
        end,
        cat(new_xs, zrs; dims = 1),
        reverse(steer_tspan(icnf, mode)),
        ps;
        icnf.sol_kwargs...,
    )
end

@views function generate_prob(
    icnf::AbstractCondICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::Any,
    n::Int,
)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    new_xs = rand_cstm_AT(icnf.resource, icnf, icnf.basedist, n)
    zrs = zeros_T_AT(icnf.resource, icnf, n_aug + 1, size(new_xs, 2))
    ϵ = randn_T_AT(icnf.resource, icnf, icnf.nvars + n_aug_input, size(new_xs, 2))
    ODEProblem{false, SciMLBase.FullSpecialize}(
        let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
            (u, p, t) -> augmented_f(u, p, t, icnf, mode, ys, ϵ, st)
        end,
        cat(new_xs, zrs; dims = 1),
        reverse(steer_tspan(icnf, mode)),
        ps;
        icnf.sol_kwargs...,
    )
end

@inline function inference(
    icnf::AbstractCondICNF,
    mode::Mode,
    xs::AbstractVecOrMat{<:Real},
    ys::AbstractVecOrMat{<:Real},
    ps::Any,
    st::Any,
)
    inference_sol(icnf, mode, inference_prob(icnf, mode, xs, ys, ps, st))
end

@inline function generate(
    icnf::AbstractCondICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any,
)
    generate_sol(icnf, mode, generate_prob(icnf, mode, ys, ps, st))
end

@inline function generate(
    icnf::AbstractCondICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::Any,
    n::Int,
)
    generate_sol(icnf, mode, generate_prob(icnf, mode, ys, ps, st, n))
end

@inline function loss(
    icnf::AbstractCondICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any,
)
    -first(inference(icnf, mode, xs, ys, ps, st))
end

@inline function loss(
    icnf::AbstractCondICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::Any,
)
    -mean(first(inference(icnf, mode, xs, ys, ps, st)))
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:ADVectorMode},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    mz, J = AbstractDifferentiation.value_and_jacobian(
        icnf.differentiation_backend,
        let ys = ys, p = p, st = st
            x -> icnf._fnn(cat(x, ys; dims = 1), p, st)
        end,
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
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    mz, J = jacobian_batched(icnf, let ys = ys, p = p, st = st
        x -> icnf._fnn(cat(x, ys; dims = 1), p, st)
    end, z)
    trace_J = transpose(tr.(eachslice(J; dims = 3)))
    cat(mz, -trace_J; dims = 1)
end

@inline function (icnf::AbstractCondICNF)(xs_ys::Any, ps::Any, st::Any)
    xs, ys = xs_ys
    first(inference(icnf, TrainMode(), xs, ys, ps, st))
end

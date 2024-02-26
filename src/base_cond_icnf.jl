export inference, generate, loss

@views function inference_prob(
    icnf::AbstractCondICNF{T, <:VectorMode, INPLACE},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any,
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    zrs = similar(xs, n_aug_input + n_aug + 1)
    @ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf.resource, icnf, icnf.nvars + n_aug_input)
    rand!(icnf.rng, icnf.epsdist, ϵ)
    ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
        ifelse(
            INPLACE,
            let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
                (du, u, p, t) -> augmented_f(du, u, p, t, icnf, mode, ys, ϵ, st)
            end,
            let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
                (u, p, t) -> augmented_f(u, p, t, icnf, mode, ys, ϵ, st)
            end,
        ),
        vcat(xs, zrs),
        steer_tspan(icnf, mode),
        ps,
    )
end

@views function inference_prob(
    icnf::AbstractCondICNF{T, <:MatrixMode, INPLACE},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::Any,
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    zrs = similar(xs, n_aug_input + n_aug + 1, size(xs, 2))
    @ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf.resource, icnf, icnf.nvars + n_aug_input, size(xs, 2))
    rand!(icnf.rng, icnf.epsdist, ϵ)
    ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
        ifelse(
            INPLACE,
            let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
                (du, u, p, t) -> augmented_f(du, u, p, t, icnf, mode, ys, ϵ, st)
            end,
            let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
                (u, p, t) -> augmented_f(u, p, t, icnf, mode, ys, ϵ, st)
            end,
        ),
        vcat(xs, zrs),
        steer_tspan(icnf, mode),
        ps,
    )
end

@views function generate_prob(
    icnf::AbstractCondICNF{T, <:VectorMode, INPLACE},
    mode::Mode,
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any,
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    new_xs = base_AT(icnf.resource, icnf, icnf.nvars + n_aug_input)
    rand!(icnf.rng, icnf.basedist, new_xs)
    zrs = similar(new_xs, n_aug + 1)
    @ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf.resource, icnf, icnf.nvars + n_aug_input)
    rand!(icnf.rng, icnf.epsdist, ϵ)
    ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
        ifelse(
            INPLACE,
            let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
                (du, u, p, t) -> augmented_f(du, u, p, t, icnf, mode, ys, ϵ, st)
            end,
            let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
                (u, p, t) -> augmented_f(u, p, t, icnf, mode, ys, ϵ, st)
            end,
        ),
        vcat(new_xs, zrs),
        reverse(steer_tspan(icnf, mode)),
        ps,
    )
end

@views function generate_prob(
    icnf::AbstractCondICNF{T, <:MatrixMode, INPLACE},
    mode::Mode,
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::Any,
    n::Int,
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    new_xs = base_AT(icnf.resource, icnf, icnf.nvars + n_aug_input, n)
    rand!(icnf.rng, icnf.basedist, new_xs)
    zrs = similar(new_xs, n_aug + 1, n)
    @ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf.resource, icnf, icnf.nvars + n_aug_input, n)
    rand!(icnf.rng, icnf.epsdist, ϵ)
    ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
        ifelse(
            INPLACE,
            let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
                (du, u, p, t) -> augmented_f(du, u, p, t, icnf, mode, ys, ϵ, st)
            end,
            let icnf = icnf, mode = mode, ys = ys, ϵ = ϵ, st = st
                (u, p, t) -> augmented_f(u, p, t, icnf, mode, ys, ϵ, st)
            end,
        ),
        vcat(new_xs, zrs),
        reverse(steer_tspan(icnf, mode)),
        ps,
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

@inline function (icnf::AbstractCondICNF)(xs_ys::Any, ps::Any, st::Any)
    xs, ys = xs_ys
    first(inference(icnf, TrainMode(), xs, ys, ps, st))
end

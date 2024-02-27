export inference, generate, loss

function inference_prob(
    icnf::AbstractICNF{T, <:VectorMode, INPLACE},
    mode::Mode,
    xs::AbstractVector{<:Real},
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
            let icnf = icnf, mode = mode, ϵ = ϵ, st = st
                (du, u, p, t) -> augmented_f(du, u, p, t, icnf, mode, ϵ, st)
            end,
            let icnf = icnf, mode = mode, ϵ = ϵ, st = st
                (u, p, t) -> augmented_f(u, p, t, icnf, mode, ϵ, st)
            end,
        ),
        vcat(xs, zrs),
        steer_tspan(icnf, mode),
        ps,
    )
end

function inference_prob(
    icnf::AbstractICNF{T, <:MatrixMode, INPLACE},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
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
            let icnf = icnf, mode = mode, ϵ = ϵ, st = st
                (du, u, p, t) -> augmented_f(du, u, p, t, icnf, mode, ϵ, st)
            end,
            let icnf = icnf, mode = mode, ϵ = ϵ, st = st
                (u, p, t) -> augmented_f(u, p, t, icnf, mode, ϵ, st)
            end,
        ),
        vcat(xs, zrs),
        steer_tspan(icnf, mode),
        ps,
    )
end

function generate_prob(
    icnf::AbstractICNF{T, <:VectorMode, INPLACE},
    mode::Mode,
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
            let icnf = icnf, mode = mode, ϵ = ϵ, st = st
                (du, u, p, t) -> augmented_f(du, u, p, t, icnf, mode, ϵ, st)
            end,
            let icnf = icnf, mode = mode, ϵ = ϵ, st = st
                (u, p, t) -> augmented_f(u, p, t, icnf, mode, ϵ, st)
            end,
        ),
        vcat(new_xs, zrs),
        reverse(steer_tspan(icnf, mode)),
        ps,
    )
end

function generate_prob(
    icnf::AbstractICNF{T, <:MatrixMode, INPLACE},
    mode::Mode,
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
            let icnf = icnf, mode = mode, ϵ = ϵ, st = st
                (du, u, p, t) -> augmented_f(du, u, p, t, icnf, mode, ϵ, st)
            end,
            let icnf = icnf, mode = mode, ϵ = ϵ, st = st
                (u, p, t) -> augmented_f(u, p, t, icnf, mode, ϵ, st)
            end,
        ),
        vcat(new_xs, zrs),
        reverse(steer_tspan(icnf, mode)),
        ps,
    )
end

@inline function inference(
    icnf::AbstractICNF,
    mode::Mode,
    xs::AbstractVecOrMat{<:Real},
    ps::Any,
    st::Any,
)
    inference_sol(icnf, mode, inference_prob(icnf, mode, xs, ps, st))
end

@inline function generate(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    ps::Any,
    st::Any,
)
    generate_sol(icnf, mode, generate_prob(icnf, mode, ps, st))
end

@inline function generate(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    ps::Any,
    st::Any,
    n::Int,
)
    generate_sol(icnf, mode, generate_prob(icnf, mode, ps, st, n))
end

@inline function loss(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::Any,
)
    -first(inference(icnf, mode, xs, ps, st))
end

@inline function loss(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::Any,
)
    -mean(first(inference(icnf, mode, xs, ps, st)))
end

@inline function (icnf::AbstractICNF)(xs::Any, ps::Any, st::Any)
    first(inference(icnf, TrainMode(), xs, ps, st))
end

function Base.show(io::IO, icnf::AbstractICNF)
    return print(io, typeof(icnf))
end

function n_augments(::AbstractICNF, ::Mode)
    return 0
end

function n_augments_input(
    icnf::AbstractICNF{<:AbstractFloat, <:ComputeMode, INPLACE, CONDITIONED, true},
) where {INPLACE, CONDITIONED}
    return icnf.naugments
end

function n_augments_input(::AbstractICNF)
    return 0
end

function steer_tspan(
    icnf::AbstractICNF{
        <:AbstractFloat,
        <:ComputeMode,
        INPLACE,
        CONDITIONED,
        AUGMENTED,
        true,
    },
    ::TrainMode{true},
) where {INPLACE, CONDITIONED, AUGMENTED}
    t₀, t₁ = icnf.tspan
    Δt = abs(t₁ - t₀)
    r = oftype(t₁, rand(icnf.rng, icnf.steerdist))
    t₁_new = muladd(Δt, r, t₁)
    return (t₀, t₁_new)
end

function steer_tspan(icnf::AbstractICNF, ::Mode)
    return icnf.tspan
end

function base_AT(icnf::AbstractICNF{T}, dims...) where {T <: AbstractFloat}
    return icnf.device(Array{T}(undef, dims...))
end

function add_conditions_nn(
    icnf::AbstractICNF{<:AbstractFloat, <:ComputeMode, INPLACE, true},
    ys::AbstractVecOrMat{<:Real},
) where {INPLACE}
    return CondLayer(icnf.nn, ys)
end

function add_conditions_nn(
    icnf::AbstractICNF{<:AbstractFloat, <:ComputeMode, INPLACE, false},
) where {INPLACE}
    return icnf.nn
end

function make_ode_func(
    icnf::AbstractICNF{T},
    mode::Mode,
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVecOrMat{T},
) where {T <: AbstractFloat}
    function ode_func(u::Any, p::Any, t::Any)
        return augmented_f(u, p, t, icnf, mode, nn, st, ϵ)
    end

    function ode_func(du::Any, u::Any, p::Any, t::Any)
        return augmented_f(du, u, p, t, icnf, mode, nn, st, ϵ)
    end

    return ode_func
end

function reg_z_aug(
    icnf::AbstractICNF{
        <:AbstractFloat,
        <:VectorMode,
        INPLACE,
        CONDITIONED,
        true,
        STEER,
        true,
    },
    ::Mode{true},
    z::Any,
) where {INPLACE, CONDITIONED, STEER}
    n_aug_input = n_augments_input(icnf)
    z_aug = z[(end - n_aug_input + 1):end]
    return LinearAlgebra.norm(z_aug)
end

function reg_z_aug(
    ::AbstractICNF{T, <:VectorMode, INPLACE, CONDITIONED, AUGMENTED, STEER, false},
    ::Mode,
    z::Any,
) where {T <: AbstractFloat, INPLACE, CONDITIONED, AUGMENTED, STEER}
    return zero(T)
end

function reg_z_aug(
    icnf::AbstractICNF{
        <:AbstractFloat,
        <:MatrixMode,
        INPLACE,
        CONDITIONED,
        true,
        STEER,
        true,
    },
    ::Mode{true},
    z::Any,
) where {INPLACE, CONDITIONED, STEER}
    n_aug_input = n_augments_input(icnf)
    z_aug = z[(end - n_aug_input + 1):end, :]
    return LinearAlgebra.norm.(eachcol(z_aug))
end

function reg_z_aug(
    ::AbstractICNF{T, <:MatrixMode, INPLACE, CONDITIONED, AUGMENTED, STEER, false},
    ::Mode,
    z::Any,
) where {T <: AbstractFloat, INPLACE, CONDITIONED, AUGMENTED, STEER}
    zrs_aug = similar(z, size(z, 2))
    ChainRulesCore.@ignore_derivatives fill!(zrs_aug, zero(T))
    return zrs_aug
end

function base_sol(
    icnf::AbstractICNF{T, <:ComputeMode, INPLACE},
    prob::SciMLBase.AbstractODEProblem{<:AbstractVecOrMat{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE}
    sol = SciMLBase.solve(prob; icnf.sol_kwargs...)
    return get_fsol(sol)
end

function inference_sol(
    icnf::AbstractICNF{T, <:VectorMode, INPLACE, CONDITIONED, AUGMENTED, STEER, NORM_Z_AUG},
    mode::Mode{REG},
    prob::SciMLBase.AbstractODEProblem{<:AbstractVector{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE, CONDITIONED, AUGMENTED, STEER, NORM_Z_AUG, REG}
    n_aug = n_augments(icnf, mode)
    fsol = base_sol(icnf, prob)
    z = fsol[begin:(end - n_aug - 1)]
    Δlogp = fsol[(end - n_aug)]
    augs = fsol[(end - n_aug + 1):end]
    logpz = oftype(Δlogp, Distributions.logpdf(icnf.basedist, z))
    logp̂x = logpz - Δlogp
    Ȧ = reg_z_aug(icnf, mode, z)
    return (logp̂x, vcat(augs, Ȧ))
end

function inference_sol(
    icnf::AbstractICNF{T, <:MatrixMode, INPLACE, CONDITIONED, AUGMENTED, STEER, NORM_Z_AUG},
    mode::Mode{REG},
    prob::SciMLBase.AbstractODEProblem{<:AbstractMatrix{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE, CONDITIONED, AUGMENTED, STEER, NORM_Z_AUG, REG}
    n_aug = n_augments(icnf, mode)
    fsol = base_sol(icnf, prob)
    z = fsol[begin:(end - n_aug - 1), :]
    Δlogp = fsol[(end - n_aug), :]
    augs = fsol[(end - n_aug + 1):end, :]
    logpz = oftype(Δlogp, Distributions.logpdf(icnf.basedist, z))
    logp̂x = logpz - Δlogp
    Ȧ = transpose(reg_z_aug(icnf, mode, z))
    return (logp̂x, eachrow(vcat(augs, Ȧ)))
end

function generate_sol(
    icnf::AbstractICNF{T, <:VectorMode, INPLACE},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractVector{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augments(icnf, mode)
    n_aug_input = n_augments_input(icnf)
    fsol = base_sol(icnf, prob)
    return fsol[begin:(end - n_aug_input - n_aug - 1)]
end

function generate_sol(
    icnf::AbstractICNF{T, <:MatrixMode, INPLACE},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractMatrix{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augments(icnf, mode)
    n_aug_input = n_augments_input(icnf)
    fsol = base_sol(icnf, prob)
    return fsol[begin:(end - n_aug_input - n_aug - 1), :]
end

function get_fsol(sol::SciMLBase.AbstractODESolution)
    return last(sol.u)
end

function get_fsol(sol::AbstractArray{T, N}) where {T, N}
    return selectdim(sol, N, lastindex(sol, N))
end

function inference_prob(
    icnf::AbstractICNF{T, <:VectorMode, INPLACE, false},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::NamedTuple,
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augments(icnf, mode)
    n_aug_input = n_augments_input(icnf)
    zrs = similar(xs, n_aug_input + n_aug + 1)
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf, icnf.nvariables + n_aug_input)
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = add_conditions_nn(icnf)
    return SciMLBase.ODEProblem{INPLACE}(
        SciMLBase.ODEFunction{INPLACE, SciMLBase.FullSpecialize}(
            make_ode_func(icnf, mode, nn, st, ϵ),
        ),
        vcat(xs, zrs),
        steer_tspan(icnf, mode),
        ps,
    )
end

function inference_prob(
    icnf::AbstractICNF{T, <:VectorMode, INPLACE, true},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::NamedTuple,
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augments(icnf, mode)
    n_aug_input = n_augments_input(icnf)
    zrs = similar(xs, n_aug_input + n_aug + 1)
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf, icnf.nvariables + n_aug_input)
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = add_conditions_nn(icnf, ys)
    return SciMLBase.ODEProblem{INPLACE}(
        SciMLBase.ODEFunction{INPLACE, SciMLBase.FullSpecialize}(
            make_ode_func(icnf, mode, nn, st, ϵ),
        ),
        vcat(xs, zrs),
        steer_tspan(icnf, mode),
        ps,
    )
end

function inference_prob(
    icnf::AbstractICNF{T, <:MatrixMode, INPLACE, false},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::NamedTuple,
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augments(icnf, mode)
    n_aug_input = n_augments_input(icnf)
    zrs = similar(xs, n_aug_input + n_aug + 1, size(xs, 2))
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf, icnf.nvariables + n_aug_input, size(xs, 2))
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = add_conditions_nn(icnf)
    return SciMLBase.ODEProblem{INPLACE}(
        SciMLBase.ODEFunction{INPLACE, SciMLBase.FullSpecialize}(
            make_ode_func(icnf, mode, nn, st, ϵ),
        ),
        vcat(xs, zrs),
        steer_tspan(icnf, mode),
        ps,
    )
end

function inference_prob(
    icnf::AbstractICNF{T, <:MatrixMode, INPLACE, true},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::NamedTuple,
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augments(icnf, mode)
    n_aug_input = n_augments_input(icnf)
    zrs = similar(xs, n_aug_input + n_aug + 1, size(xs, 2))
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf, icnf.nvariables + n_aug_input, size(xs, 2))
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = add_conditions_nn(icnf, ys)
    return SciMLBase.ODEProblem{INPLACE}(
        SciMLBase.ODEFunction{INPLACE, SciMLBase.FullSpecialize}(
            make_ode_func(icnf, mode, nn, st, ϵ),
        ),
        vcat(xs, zrs),
        steer_tspan(icnf, mode),
        ps,
    )
end

function generate_prob(
    icnf::AbstractICNF{T, <:VectorMode, INPLACE, false},
    mode::Mode,
    ps::Any,
    st::NamedTuple,
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augments(icnf, mode)
    n_aug_input = n_augments_input(icnf)
    new_xs = base_AT(icnf, icnf.nvariables + n_aug_input)
    Random.rand!(icnf.rng, icnf.basedist, new_xs)
    zrs = similar(new_xs, n_aug + 1)
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf, icnf.nvariables + n_aug_input)
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = add_conditions_nn(icnf)
    return SciMLBase.ODEProblem{INPLACE}(
        SciMLBase.ODEFunction{INPLACE, SciMLBase.FullSpecialize}(
            make_ode_func(icnf, mode, nn, st, ϵ),
        ),
        vcat(new_xs, zrs),
        reverse(steer_tspan(icnf, mode)),
        ps,
    )
end

function generate_prob(
    icnf::AbstractICNF{T, <:VectorMode, INPLACE, true},
    mode::Mode,
    ys::AbstractVector{<:Real},
    ps::Any,
    st::NamedTuple,
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augments(icnf, mode)
    n_aug_input = n_augments_input(icnf)
    new_xs = base_AT(icnf, icnf.nvariables + n_aug_input)
    Random.rand!(icnf.rng, icnf.basedist, new_xs)
    zrs = similar(new_xs, n_aug + 1)
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf, icnf.nvariables + n_aug_input)
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = add_conditions_nn(icnf, ys)
    return SciMLBase.ODEProblem{INPLACE}(
        SciMLBase.ODEFunction{INPLACE, SciMLBase.FullSpecialize}(
            make_ode_func(icnf, mode, nn, st, ϵ),
        ),
        vcat(new_xs, zrs),
        reverse(steer_tspan(icnf, mode)),
        ps,
    )
end

function generate_prob(
    icnf::AbstractICNF{T, <:MatrixMode, INPLACE, false},
    mode::Mode,
    ps::Any,
    st::NamedTuple,
    n::Int,
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augments(icnf, mode)
    n_aug_input = n_augments_input(icnf)
    new_xs = base_AT(icnf, icnf.nvariables + n_aug_input, n)
    Random.rand!(icnf.rng, icnf.basedist, new_xs)
    zrs = similar(new_xs, n_aug + 1, n)
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf, icnf.nvariables + n_aug_input, n)
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = add_conditions_nn(icnf)
    return SciMLBase.ODEProblem{INPLACE}(
        SciMLBase.ODEFunction{INPLACE, SciMLBase.FullSpecialize}(
            make_ode_func(icnf, mode, nn, st, ϵ),
        ),
        vcat(new_xs, zrs),
        reverse(steer_tspan(icnf, mode)),
        ps,
    )
end

function generate_prob(
    icnf::AbstractICNF{T, <:MatrixMode, INPLACE, true},
    mode::Mode,
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::NamedTuple,
    n::Int,
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augments(icnf, mode)
    n_aug_input = n_augments_input(icnf)
    new_xs = base_AT(icnf, icnf.nvariables + n_aug_input, n)
    Random.rand!(icnf.rng, icnf.basedist, new_xs)
    zrs = similar(new_xs, n_aug + 1, n)
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf, icnf.nvariables + n_aug_input, n)
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = add_conditions_nn(icnf, ys)
    return SciMLBase.ODEProblem{INPLACE}(
        SciMLBase.ODEFunction{INPLACE, SciMLBase.FullSpecialize}(
            make_ode_func(icnf, mode, nn, st, ϵ),
        ),
        vcat(new_xs, zrs),
        reverse(steer_tspan(icnf, mode)),
        ps,
    )
end

function inference(
    icnf::AbstractICNF,
    mode::Mode,
    xs::AbstractVecOrMat{<:Real},
    ps::Any,
    st::NamedTuple,
)
    return inference_sol(icnf, mode, inference_prob(icnf, mode, xs, ps, st))
end

function inference(
    icnf::AbstractICNF,
    mode::Mode,
    xs::AbstractVecOrMat{<:Real},
    ys::AbstractVecOrMat{<:Real},
    ps::Any,
    st::NamedTuple,
)
    return inference_sol(icnf, mode, inference_prob(icnf, mode, xs, ys, ps, st))
end

function generate(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    ps::Any,
    st::NamedTuple,
)
    return generate_sol(icnf, mode, generate_prob(icnf, mode, ps, st))
end

function generate(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    ys::AbstractVector{<:Real},
    ps::Any,
    st::NamedTuple,
)
    return generate_sol(icnf, mode, generate_prob(icnf, mode, ys, ps, st))
end

function generate(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    ps::Any,
    st::NamedTuple,
    n::Int,
)
    return generate_sol(icnf, mode, generate_prob(icnf, mode, ps, st, n))
end

function generate(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::NamedTuple,
    n::Int,
)
    return generate_sol(icnf, mode, generate_prob(icnf, mode, ys, ps, st, n))
end

function loss(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::NamedTuple,
)
    return -first(inference(icnf, mode, xs, ps, st))
end

function loss(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::NamedTuple,
)
    return -first(inference(icnf, mode, xs, ys, ps, st))
end

function loss(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::NamedTuple,
)
    return -Statistics.mean(first(inference(icnf, mode, xs, ps, st)))
end

function loss(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::NamedTuple,
)
    return -Statistics.mean(first(inference(icnf, mode, xs, ys, ps, st)))
end

function (icnf::AbstractICNF{<:AbstractFloat, <:ComputeMode, INPLACE, false})(
    xs::AbstractVecOrMat{<:Real},
    ps::Any,
    st::NamedTuple,
) where {INPLACE}
    return first(inference(icnf, TrainMode{false}(), xs, ps, st)), st
end

function (icnf::AbstractICNF{<:AbstractFloat, <:ComputeMode, INPLACE, true})(
    (xs, ys)::Tuple{<:AbstractVecOrMat{<:Real}, <:AbstractVecOrMat{<:Real}},
    ps::Any,
    st::NamedTuple,
) where {INPLACE}
    return first(inference(icnf, TrainMode{false}(), xs, ys, ps, st)), st
end

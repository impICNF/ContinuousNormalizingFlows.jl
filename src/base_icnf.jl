function construct(
    aicnf::Type{<:AbstractICNF},
    nn::LuxCore.AbstractLuxLayer,
    nvars::Int,
    naugmented::Int = 0;
    data_type::Type{<:AbstractFloat} = Float32,
    compute_mode::ComputeMode = LuxVecJacMatrixMode(ADTypes.AutoZygote()),
    inplace::Bool = false,
    cond::Bool = aicnf <: Union{CondRNODE, CondFFJORD, CondPlanar},
    device::MLDataDevices.AbstractDevice = MLDataDevices.cpu_device(),
    basedist::Distributions.Distribution = Distributions.MvNormal(
        FillArrays.Zeros{data_type}(nvars + naugmented),
        FillArrays.Eye{data_type}(nvars + naugmented),
    ),
    tspan::NTuple{2} = (zero(data_type), one(data_type)),
    steer_rate::AbstractFloat = zero(data_type),
    epsdist::Distributions.Distribution = Distributions.MvNormal(
        FillArrays.Zeros{data_type}(nvars + naugmented),
        FillArrays.Eye{data_type}(nvars + naugmented),
    ),
    sol_kwargs::NamedTuple = (;
        progress = true,
        save_everystep = false,
        reltol = sqrt(eps(one(data_type))),
        abstol = eps(one(data_type)),
        maxiters = typemax(Int),
        alg = OrdinaryDiffEqDefault.DefaultODEAlgorithm(),
    ),
    rng::Random.AbstractRNG = MLDataDevices.default_device_rng(device),
    λ₁::AbstractFloat = if aicnf <: Union{RNODE, CondRNODE}
        convert(data_type, 1.0e-2)
    else
        zero(data_type)
    end,
    λ₂::AbstractFloat = if aicnf <: Union{RNODE, CondRNODE}
        convert(data_type, 1.0e-2)
    else
        zero(data_type)
    end,
    λ₃::AbstractFloat = if naugmented >= nvars
        convert(data_type, 1.0e-2)
    else
        zero(data_type)
    end,
)
    steerdist = Distributions.Uniform{data_type}(-steer_rate, steer_rate)

    return ICNF{
        data_type,
        typeof(compute_mode),
        inplace,
        cond,
        !iszero(naugmented),
        !iszero(steer_rate),
        !iszero(λ₁),
        !iszero(λ₂),
        !iszero(λ₃),
        typeof(nn),
        typeof(nvars),
        typeof(device),
        typeof(basedist),
        typeof(tspan),
        typeof(steerdist),
        typeof(epsdist),
        typeof(sol_kwargs),
        typeof(rng),
    }(
        nn,
        nvars,
        naugmented,
        compute_mode,
        device,
        basedist,
        tspan,
        steerdist,
        epsdist,
        sol_kwargs,
        rng,
        λ₁,
        λ₂,
        λ₃,
    )
end

function Base.show(io::IO, icnf::AbstractICNF)
    return print(io, typeof(icnf))
end

@inline function n_augment(::AbstractICNF, ::Mode)
    return 0
end

@inline function n_augment_input(
    icnf::AbstractICNF{<:AbstractFloat, <:ComputeMode, INPLACE, COND, true},
) where {INPLACE, COND}
    return icnf.naugmented
end

@inline function n_augment_input(::AbstractICNF)
    return 0
end

@inline function steer_tspan(
    icnf::AbstractICNF{T, <:ComputeMode, INPLACE, COND, AUGMENTED, true},
    ::TrainMode,
) where {T <: AbstractFloat, INPLACE, COND, AUGMENTED}
    t₀, t₁ = icnf.tspan
    Δt = abs(t₁ - t₀)
    r = convert(T, rand(icnf.rng, icnf.steerdist))
    t₁_new = muladd(Δt, r, t₁)
    return (t₀, t₁_new)
end

@inline function steer_tspan(icnf::AbstractICNF, ::Mode)
    return icnf.tspan
end

@inline function base_AT(icnf::AbstractICNF{T}, dims...) where {T <: AbstractFloat}
    return icnf.device(Array{T}(undef, dims...))
end

ChainRulesCore.@non_differentiable base_AT(::Any...)

function base_sol(
    icnf::AbstractICNF{T, <:ComputeMode, INPLACE},
    prob::SciMLBase.AbstractODEProblem{<:AbstractVecOrMat{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE}
    sol = SciMLBase.solve(prob; icnf.sol_kwargs...)
    return get_fsol(sol)
end

function inference_sol(
    icnf::AbstractICNF{T, <:VectorMode, INPLACE, COND, AUGMENTED, STEER, NORM_Z_AUG},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractVector{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE, COND, AUGMENTED, STEER, NORM_Z_AUG}
    n_aug = n_augment(icnf, mode)
    fsol = base_sol(icnf, prob)
    z = fsol[begin:(end - n_aug - 1)]
    Δlogp = fsol[(end - n_aug)]
    augs = fsol[(end - n_aug + 1):end]
    logpz = oftype(Δlogp, Distributions.logpdf(icnf.basedist, z))
    logp̂x = logpz - Δlogp
    Ȧ = if (NORM_Z_AUG && AUGMENTED)
        n_aug_input = n_augment_input(icnf)
        z_aug = z[(end - n_aug_input + 1):end]
        LinearAlgebra.norm(z_aug)
    else
        zero(T)
    end
    return (logp̂x, vcat(augs, Ȧ))
end

function inference_sol(
    icnf::AbstractICNF{T, <:MatrixMode, INPLACE, COND, AUGMENTED, STEER, NORM_Z_AUG},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractMatrix{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE, COND, AUGMENTED, STEER, NORM_Z_AUG}
    n_aug = n_augment(icnf, mode)
    fsol = base_sol(icnf, prob)
    z = fsol[begin:(end - n_aug - 1), :]
    Δlogp = fsol[(end - n_aug), :]
    augs = fsol[(end - n_aug + 1):end, :]
    logpz = oftype(Δlogp, Distributions.logpdf(icnf.basedist, z))
    logp̂x = logpz - Δlogp
    Ȧ = transpose(if (NORM_Z_AUG && AUGMENTED)
        n_aug_input = n_augment_input(icnf)
        z_aug = z[(end - n_aug_input + 1):end, :]
        LinearAlgebra.norm.(eachcol(z_aug))
    else
        zrs_aug = similar(augs, size(augs, 2))
        ChainRulesCore.@ignore_derivatives fill!(zrs_aug, zero(T))
        zrs_aug
    end)
    return (logp̂x, eachrow(vcat(augs, Ȧ)))
end

function generate_sol(
    icnf::AbstractICNF{T, <:VectorMode, INPLACE},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractVector{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    fsol = base_sol(icnf, prob)
    return fsol[begin:(end - n_aug_input - n_aug - 1)]
end

function generate_sol(
    icnf::AbstractICNF{T, <:MatrixMode, INPLACE},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractMatrix{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    fsol = base_sol(icnf, prob)
    return fsol[begin:(end - n_aug_input - n_aug - 1), :]
end

@inline function get_fsol(sol::SciMLBase.AbstractODESolution)
    return last(sol.u)
end

@inline function get_fsol(sol::AbstractArray{T, N}) where {T, N}
    return selectdim(sol, N, lastindex(sol, N))
end

function inference_prob(
    icnf::AbstractICNF{T, <:VectorMode, INPLACE, false},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::NamedTuple,
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    zrs = similar(xs, n_aug_input + n_aug + 1)
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf, icnf.nvars + n_aug_input)
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = icnf.nn
    return SciMLBase.ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
        make_ode_func(icnf, mode, nn, st, ϵ),
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
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    zrs = similar(xs, n_aug_input + n_aug + 1)
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf, icnf.nvars + n_aug_input)
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = CondLayer(icnf.nn, ys)
    return SciMLBase.ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
        make_ode_func(icnf, mode, nn, st, ϵ),
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
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    zrs = similar(xs, n_aug_input + n_aug + 1, size(xs, 2))
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf, icnf.nvars + n_aug_input, size(xs, 2))
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = icnf.nn
    return SciMLBase.ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
        make_ode_func(icnf, mode, nn, st, ϵ),
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
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    zrs = similar(xs, n_aug_input + n_aug + 1, size(xs, 2))
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf, icnf.nvars + n_aug_input, size(xs, 2))
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = CondLayer(icnf.nn, ys)
    return SciMLBase.ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
        make_ode_func(icnf, mode, nn, st, ϵ),
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
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    new_xs = base_AT(icnf, icnf.nvars + n_aug_input)
    Random.rand!(icnf.rng, icnf.basedist, new_xs)
    zrs = similar(new_xs, n_aug + 1)
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf, icnf.nvars + n_aug_input)
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = icnf.nn
    return SciMLBase.ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
        make_ode_func(icnf, mode, nn, st, ϵ),
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
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    new_xs = base_AT(icnf, icnf.nvars + n_aug_input)
    Random.rand!(icnf.rng, icnf.basedist, new_xs)
    zrs = similar(new_xs, n_aug + 1)
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf, icnf.nvars + n_aug_input)
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = CondLayer(icnf.nn, ys)
    return SciMLBase.ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
        make_ode_func(icnf, mode, nn, st, ϵ),
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
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    new_xs = base_AT(icnf, icnf.nvars + n_aug_input, n)
    Random.rand!(icnf.rng, icnf.basedist, new_xs)
    zrs = similar(new_xs, n_aug + 1, n)
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf, icnf.nvars + n_aug_input, n)
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = icnf.nn
    return SciMLBase.ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
        make_ode_func(icnf, mode, nn, st, ϵ),
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
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    new_xs = base_AT(icnf, icnf.nvars + n_aug_input, n)
    Random.rand!(icnf.rng, icnf.basedist, new_xs)
    zrs = similar(new_xs, n_aug + 1, n)
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf, icnf.nvars + n_aug_input, n)
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = CondLayer(icnf.nn, ys)
    return SciMLBase.ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
        make_ode_func(icnf, mode, nn, st, ϵ),
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
    st::NamedTuple,
)
    return inference_sol(icnf, mode, inference_prob(icnf, mode, xs, ps, st))
end

@inline function inference(
    icnf::AbstractICNF,
    mode::Mode,
    xs::AbstractVecOrMat{<:Real},
    ys::AbstractVecOrMat{<:Real},
    ps::Any,
    st::NamedTuple,
)
    return inference_sol(icnf, mode, inference_prob(icnf, mode, xs, ys, ps, st))
end

@inline function generate(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    ps::Any,
    st::NamedTuple,
)
    return generate_sol(icnf, mode, generate_prob(icnf, mode, ps, st))
end

@inline function generate(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    ys::AbstractVector{<:Real},
    ps::Any,
    st::NamedTuple,
)
    return generate_sol(icnf, mode, generate_prob(icnf, mode, ys, ps, st))
end

@inline function generate(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    ps::Any,
    st::NamedTuple,
    n::Int,
)
    return generate_sol(icnf, mode, generate_prob(icnf, mode, ps, st, n))
end

@inline function generate(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::NamedTuple,
    n::Int,
)
    return generate_sol(icnf, mode, generate_prob(icnf, mode, ys, ps, st, n))
end

@inline function loss(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::NamedTuple,
)
    return -first(inference(icnf, mode, xs, ps, st))
end

@inline function loss(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::NamedTuple,
)
    return -first(inference(icnf, mode, xs, ys, ps, st))
end

@inline function loss(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::NamedTuple,
)
    return -Statistics.mean(first(inference(icnf, mode, xs, ps, st)))
end

@inline function loss(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::NamedTuple,
)
    return -Statistics.mean(first(inference(icnf, mode, xs, ys, ps, st)))
end

@inline function make_ode_func(
    icnf::AbstractICNF{T, CM, INPLACE},
    mode::Mode,
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVecOrMat{T},
) where {T <: AbstractFloat, CM, INPLACE}
    function ode_func_op(u, p, t)
        return augmented_f(u, p, t, icnf, mode, nn, st, ϵ)
    end

    function ode_func_ip(du, u, p, t)
        return augmented_f(du, u, p, t, icnf, mode, nn, st, ϵ)
    end

    return ifelse(INPLACE, ode_func_ip, ode_func_op)
end

@inline function (icnf::AbstractICNF{T, CM, INPLACE, false})(
    xs::AbstractVecOrMat,
    ps::Any,
    st::NamedTuple,
) where {T, CM, INPLACE}
    return first(inference(icnf, TrainMode(), xs, ps, st)), st
end

@inline function (icnf::AbstractICNF{T, CM, INPLACE, true})(
    xs_ys::Tuple,
    ps::Any,
    st::NamedTuple,
) where {T, CM, INPLACE}
    xs, ys = xs_ys
    return first(inference(icnf, TrainMode(), xs, ys, ps, st)), st
end

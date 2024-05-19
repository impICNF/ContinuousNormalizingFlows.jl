function construct(
    aicnf::Type{<:AbstractICNF},
    nn::LuxCore.AbstractExplicitLayer,
    nvars::Int,
    naugmented::Int = 0;
    data_type::Type{<:AbstractFloat} = Float32,
    compute_mode::Type{<:ComputeMode} = ADVecJacVectorMode,
    inplace::Bool = false,
    cond::Bool = aicnf <: Union{CondRNODE, CondFFJORD, CondPlanar},
    resource::ComputationalResources.AbstractResource = ComputationalResources.CPU1(),
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
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    autodiff_backend::ADTypes.AbstractADType = ADTypes.AutoZygote(),
    sol_kwargs::NamedTuple = (
        save_everystep = false,
        alg = OrdinaryDiffEq.Tsit5(; thread = Static.True()),
    ),
    rng::Random.AbstractRNG = rng_AT(resource),
    λ₁::AbstractFloat = if aicnf <: Union{RNODE, CondRNODE}
        convert(data_type, 1e-2)
    else
        zero(data_type)
    end,
    λ₂::AbstractFloat = if aicnf <: Union{RNODE, CondRNODE}
        convert(data_type, 1e-2)
    else
        zero(data_type)
    end,
    λ₃::AbstractFloat = zero(data_type),
)
    steerdist = Distributions.Uniform{data_type}(-steer_rate, steer_rate)

    ICNF{
        data_type,
        compute_mode,
        inplace,
        cond,
        !iszero(naugmented),
        !iszero(steer_rate),
        !iszero(λ₁),
        !iszero(λ₂),
        !iszero(λ₃),
        typeof(nn),
        typeof(nvars),
        typeof(resource),
        typeof(basedist),
        typeof(tspan),
        typeof(steerdist),
        typeof(epsdist),
        typeof(differentiation_backend),
        typeof(autodiff_backend),
        typeof(sol_kwargs),
        typeof(rng),
    }(
        nn,
        nvars,
        naugmented,
        resource,
        basedist,
        tspan,
        steerdist,
        epsdist,
        differentiation_backend,
        autodiff_backend,
        sol_kwargs,
        rng,
        λ₁,
        λ₂,
        λ₃,
    )
end

@inline function n_augment(::AbstractICNF, ::Mode)
    0
end

function Base.show(io::IO, icnf::AbstractICNF)
    print(
        io,
        typeof(icnf),
        "<",
        "Number of Variables: ",
        icnf.nvars,
        ", Number of Augmentations: ",
        n_augment_input(icnf),
        ", Time Span: ",
        icnf.tspan,
        ">",
    )
end

@inline function n_augment_input(
    icnf::AbstractICNF{<:AbstractFloat, <:ComputeMode, INPLACE, COND, true},
) where {INPLACE, COND}
    icnf.naugmented
end

@inline function n_augment_input(::AbstractICNF)
    0
end

@inline function steer_tspan(
    icnf::AbstractICNF{T, <:ComputeMode, INPLACE, COND, AUGMENTED, true},
    ::TrainMode,
) where {T <: AbstractFloat, INPLACE, COND, AUGMENTED}
    t₀, t₁ = icnf.tspan
    Δt = abs(t₁ - t₀)
    r = convert(T, rand(icnf.rng, icnf.steerdist))
    t₁_new = muladd(Δt, r, t₁)
    (t₀, t₁_new)
end

@inline function steer_tspan(icnf::AbstractICNF, ::Mode)
    icnf.tspan
end

@inline function rng_AT(::ComputationalResources.AbstractResource)
    Random.default_rng()
end

@inline function base_AT(
    ::ComputationalResources.AbstractResource,
    ::AbstractICNF{T},
    dims...,
) where {T <: AbstractFloat}
    Array{T}(undef, dims...)
end

ChainRulesCore.@non_differentiable base_AT(::Any...)

function base_sol(
    icnf::AbstractICNF{T, <:ComputeMode, INPLACE},
    prob::SciMLBase.AbstractODEProblem{<:AbstractVecOrMat{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE}
    sol = SciMLBase.solve(prob; icnf.sol_kwargs...)
    get_fsol(sol)
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
    (logp̂x, vcat(augs, Ȧ))
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
    (logp̂x, eachrow(vcat(augs, Ȧ)))
end

function generate_sol(
    icnf::AbstractICNF{T, <:VectorMode, INPLACE},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractVector{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    fsol = base_sol(icnf, prob)
    fsol[begin:(end - n_aug_input - n_aug - 1)]
end

function generate_sol(
    icnf::AbstractICNF{T, <:MatrixMode, INPLACE},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractMatrix{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    fsol = base_sol(icnf, prob)
    fsol[begin:(end - n_aug_input - n_aug - 1), :]
end

@inline function get_fsol(sol::SciMLBase.AbstractODESolution)
    last(sol.u)
end

@inline function get_fsol(sol::AbstractArray{T, N}) where {T, N}
    selectdim(sol, N, lastindex(sol, N))
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
    ϵ = base_AT(icnf.resource, icnf, icnf.nvars + n_aug_input)
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = icnf.nn
    SciMLBase.ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
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
    ϵ = base_AT(icnf.resource, icnf, icnf.nvars + n_aug_input)
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = CondLayer(icnf.nn, ys)
    SciMLBase.ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
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
    ϵ = base_AT(icnf.resource, icnf, icnf.nvars + n_aug_input, size(xs, 2))
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = icnf.nn
    SciMLBase.ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
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
    ϵ = base_AT(icnf.resource, icnf, icnf.nvars + n_aug_input, size(xs, 2))
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = CondLayer(icnf.nn, ys)
    SciMLBase.ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
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
    new_xs = base_AT(icnf.resource, icnf, icnf.nvars + n_aug_input)
    Random.rand!(icnf.rng, icnf.basedist, new_xs)
    zrs = similar(new_xs, n_aug + 1)
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf.resource, icnf, icnf.nvars + n_aug_input)
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = icnf.nn
    SciMLBase.ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
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
    new_xs = base_AT(icnf.resource, icnf, icnf.nvars + n_aug_input)
    Random.rand!(icnf.rng, icnf.basedist, new_xs)
    zrs = similar(new_xs, n_aug + 1)
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf.resource, icnf, icnf.nvars + n_aug_input)
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = CondLayer(icnf.nn, ys)
    SciMLBase.ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
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
    new_xs = base_AT(icnf.resource, icnf, icnf.nvars + n_aug_input, n)
    Random.rand!(icnf.rng, icnf.basedist, new_xs)
    zrs = similar(new_xs, n_aug + 1, n)
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf.resource, icnf, icnf.nvars + n_aug_input, n)
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = icnf.nn
    SciMLBase.ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
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
    new_xs = base_AT(icnf.resource, icnf, icnf.nvars + n_aug_input, n)
    Random.rand!(icnf.rng, icnf.basedist, new_xs)
    zrs = similar(new_xs, n_aug + 1, n)
    ChainRulesCore.@ignore_derivatives fill!(zrs, zero(T))
    ϵ = base_AT(icnf.resource, icnf, icnf.nvars + n_aug_input, n)
    Random.rand!(icnf.rng, icnf.epsdist, ϵ)
    nn = CondLayer(icnf.nn, ys)
    SciMLBase.ODEProblem{INPLACE, SciMLBase.FullSpecialize}(
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
    inference_sol(icnf, mode, inference_prob(icnf, mode, xs, ps, st))
end

@inline function inference(
    icnf::AbstractICNF,
    mode::Mode,
    xs::AbstractVecOrMat{<:Real},
    ys::AbstractVecOrMat{<:Real},
    ps::Any,
    st::NamedTuple,
)
    inference_sol(icnf, mode, inference_prob(icnf, mode, xs, ys, ps, st))
end

@inline function generate(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    ps::Any,
    st::NamedTuple,
)
    generate_sol(icnf, mode, generate_prob(icnf, mode, ps, st))
end

@inline function generate(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    ys::AbstractVector{<:Real},
    ps::Any,
    st::NamedTuple,
)
    generate_sol(icnf, mode, generate_prob(icnf, mode, ys, ps, st))
end

@inline function generate(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    ps::Any,
    st::NamedTuple,
    n::Int,
)
    generate_sol(icnf, mode, generate_prob(icnf, mode, ps, st, n))
end

@inline function generate(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::NamedTuple,
    n::Int,
)
    generate_sol(icnf, mode, generate_prob(icnf, mode, ys, ps, st, n))
end

@inline function loss(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::NamedTuple,
)
    -first(inference(icnf, mode, xs, ps, st))
end

@inline function loss(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::NamedTuple,
)
    -first(inference(icnf, mode, xs, ys, ps, st))
end

@inline function loss(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::NamedTuple,
)
    -Statistics.mean(first(inference(icnf, mode, xs, ps, st)))
end

@inline function loss(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::NamedTuple,
)
    -Statistics.mean(first(inference(icnf, mode, xs, ys, ps, st)))
end

@inline function make_ode_func(
    icnf::AbstractICNF{T, CM, INPLACE},
    mode::Mode,
    nn::LuxCore.AbstractExplicitLayer,
    st::NamedTuple,
    ϵ::AbstractVecOrMat{T},
) where {T <: AbstractFloat, CM, INPLACE}
    function ode_func_op(u, p, t)
        augmented_f(u, p, t, icnf, mode, nn, st, ϵ)
    end

    function ode_func_ip(du, u, p, t)
        augmented_f(du, u, p, t, icnf, mode, nn, st, ϵ)
    end

    ifelse(INPLACE, ode_func_ip, ode_func_op)
end

@inline function (icnf::AbstractICNF{T, CM, INPLACE, false})(
    xs::AbstractVecOrMat,
    ps::Any,
    st::NamedTuple,
) where {T, CM, INPLACE}
    first(inference(icnf, TrainMode(), xs, ps, st)), st
end

@inline function (icnf::AbstractICNF{T, CM, INPLACE, true})(
    xs_ys::Tuple,
    ps::Any,
    st::NamedTuple,
) where {T, CM, INPLACE}
    xs, ys = xs_ys
    first(inference(icnf, TrainMode(), xs, ys, ps, st)), st
end

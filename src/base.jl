export construct

function construct(
    aicnf::Type{<:AbstractFlows},
    nn,
    nvars::Int,
    naugmented::Int = 0;
    data_type::Type{<:AbstractFloat} = Float32,
    compute_mode::Type{<:ComputeMode} = ADVecJacVectorMode,
    inplace::Bool = false,
    resource::AbstractResource = CPU1(),
    basedist::Distribution = MvNormal(
        Zeros{data_type}(nvars + naugmented),
        Eye{data_type}(nvars + naugmented),
    ),
    tspan::NTuple{2} = (zero(data_type), one(data_type)),
    steer_rate::AbstractFloat = zero(data_type),
    epsdist::Distribution = MvNormal(
        Zeros{data_type}(nvars + naugmented),
        Eye{data_type}(nvars + naugmented),
    ),
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    autodiff_backend::ADTypes.AbstractADType = ifelse(
        compute_mode <: SDJacVecMatrixMode,
        AutoForwardDiff(),
        AutoZygote(),
    ),
    sol_kwargs::NamedTuple = sol_kwargs_defaults.medium,
    rng::AbstractRNG = rng_AT(resource),
    λ₁::AbstractFloat = convert(data_type, 1e-2),
    λ₂::AbstractFloat = convert(data_type, 1e-2),
)
    steerdist = Uniform{data_type}(-steer_rate, steer_rate)

    if aicnf <: Union{RNODE, CondRNODE}
        aicnf{
            data_type,
            compute_mode,
            inplace,
            !iszero(naugmented),
            !iszero(steer_rate),
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
        )
    else
        aicnf{
            data_type,
            compute_mode,
            inplace,
            !iszero(naugmented),
            !iszero(steer_rate),
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
        )
    end
end

@inline function n_augment(::AbstractFlows, ::Mode)
    0
end

# pretty-printing

function Base.show(io::IO, icnf::AbstractFlows)
    print(
        io,
        typeof(icnf),
        "\n\tNumber of Variables: ",
        icnf.nvars,
        "\n\tNumber of Augmentations: ",
        n_augment_input(icnf),
        "\n\tTime Span: ",
        icnf.tspan,
    )
end

@inline function n_augment_input(
    icnf::AbstractFlows{<:AbstractFloat, <:ComputeMode, INPLACE, true},
) where {INPLACE}
    icnf.naugmented
end

@inline function n_augment_input(::AbstractFlows)
    0
end

@inline function steer_tspan(
    icnf::AbstractFlows{T, <:ComputeMode, INPLACE, AUGMENTED, true},
    ::TrainMode,
) where {T <: AbstractFloat, INPLACE, AUGMENTED}
    t₀, t₁ = icnf.tspan
    Δt = abs(t₁ - t₀)
    r = convert(T, rand(icnf.rng, icnf.steerdist))
    t₁_new = muladd(Δt, r, t₁)
    (t₀, t₁_new)
end

@inline function steer_tspan(icnf::AbstractFlows, ::Mode)
    icnf.tspan
end

@inline function rng_AT(::AbstractResource)
    Random.default_rng()
end

@inline function base_AT(::AbstractResource)
    Array
end

function inference_sol(
    icnf::AbstractFlows{T, <:VectorMode, INPLACE},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractVector{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    sol = solve(prob; icnf.sol_kwargs...)
    fsol = get_fsol(sol)
    z = fsol[begin:(end - n_aug - 1)]
    Δlogp = fsol[(end - n_aug)]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    augs = fsol[(end - n_aug + 1):end]
    (logp̂x, augs)
end

function inference_sol(
    icnf::AbstractFlows{T, <:MatrixMode, INPLACE},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractMatrix{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    sol = solve(prob; icnf.sol_kwargs...)
    fsol = get_fsol(sol)
    z = fsol[begin:(end - n_aug - 1), :]
    Δlogp = fsol[(end - n_aug), :]
    logp̂x = logpdf(icnf.basedist, z) - Δlogp
    augs = fsol[(end - n_aug + 1):end, :]
    (logp̂x, eachrow(augs))
end

function generate_sol(
    icnf::AbstractFlows{T, <:VectorMode, INPLACE},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractVector{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    sol = solve(prob; icnf.sol_kwargs...)
    fsol = get_fsol(sol)
    z = fsol[begin:(end - n_aug_input - n_aug - 1)]
    z
end

function generate_sol(
    icnf::AbstractFlows{T, <:MatrixMode, INPLACE},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractMatrix{<:Real}, NTuple{2, T}, INPLACE},
) where {T <: AbstractFloat, INPLACE}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    sol = solve(prob; icnf.sol_kwargs...)
    fsol = get_fsol(sol)
    z = fsol[begin:(end - n_aug_input - n_aug - 1), :]
    z
end

@inline function get_fsol(sol::SciMLBase.AbstractODESolution)
    last(sol.u)
end

@inline function get_fsol(sol::AbstractArray{T, N}) where {T, N}
    selectdim(sol, N, lastindex(sol, N))
end

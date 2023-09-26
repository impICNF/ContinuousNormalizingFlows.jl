export construct

function construct(
    aicnf::Type{<:AbstractFlows},
    nn,
    nvars::Int,
    naugmented::Int = 0;
    data_type::Type{<:AbstractFloat} = Float32,
    compute_mode::Type{<:ComputeMode} = ADVectorMode,
    resource::AbstractResource = CPU1(),
    basedist::Distribution = MvNormal(
        Zeros{data_type}(nvars + naugmented),
        Eye{data_type}(nvars + naugmented),
    ),
    tspan::NTuple{2} = (zero(data_type), one(data_type)),
    steer_rate::AbstractFloat = zero(data_type),
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    autodiff_backend::ADTypes.AbstractADType = AutoZygote(),
    sol_kwargs::Dict = Dict(
        :alg_hints => [:nonstiff, :memorybound],
        :save_everystep => false,
        :alg => VCABM(),
        :sensealg => InterpolatingAdjoint(;
            autodiff = true,
            autojacvec = ZygoteVJP(),
            checkpointing = true,
        ),
        :reltol => sqrt(eps(one(Float32))),
        :abstol => eps(one(Float32)),
        :maxiters => typemax(Int32),
    ),
    rng::AbstractRNG = Random.default_rng(),
)
    steerdist = Uniform{data_type}(-steer_rate, steer_rate)
    _fnn(x, ps, st) = first(nn(x, ps, st))

    aicnf{
        data_type,
        compute_mode,
        !iszero(naugmented),
        !iszero(steer_rate),
        typeof(nn),
        typeof(nvars),
        typeof(resource),
        typeof(basedist),
        typeof(tspan),
        typeof(steerdist),
        typeof(differentiation_backend),
        typeof(autodiff_backend),
        typeof(sol_kwargs),
        typeof(rng),
        typeof(_fnn),
    }(
        nn,
        nvars,
        naugmented,
        resource,
        basedist,
        tspan,
        steerdist,
        differentiation_backend,
        autodiff_backend,
        sol_kwargs,
        rng,
        _fnn,
    )
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

@inline function n_augment_input(icnf::AbstractFlows{<:AbstractFloat, <:ComputeMode, true})
    icnf.naugmented
end

@inline function n_augment_input(::AbstractFlows)
    0
end

@inline function steer_tspan(
    icnf::AbstractFlows{<:AbstractFloat, <:ComputeMode, AUGMENTED, true},
    ::TrainMode,
) where {AUGMENTED}
    t₀, t₁ = icnf.tspan
    Δt = abs(t₁ - t₀)
    r = rand_cstm_AT(icnf.resource, icnf, icnf.steerdist)
    t₁_new = muladd(Δt, r, t₁)
    (t₀, t₁_new)
end

@inline function steer_tspan(icnf::AbstractFlows, ::Mode)
    icnf.tspan
end

@inline function zeros_T_AT(
    ::AbstractResource,
    ::AbstractFlows{T},
    dims...,
) where {T <: AbstractFloat}
    zeros(T, dims...)
end

@inline function rand_T_AT(
    ::AbstractResource,
    icnf::AbstractFlows{T},
    dims...,
) where {T <: AbstractFloat}
    rand(icnf.rng, T, dims...)
end

@inline function randn_T_AT(
    ::AbstractResource,
    icnf::AbstractFlows{T},
    dims...,
) where {T <: AbstractFloat}
    randn(icnf.rng, T, dims...)
end

@inline function rand_cstm_AT(
    ::AbstractResource,
    icnf::AbstractFlows{T},
    cstm::Any,
    dims...,
) where {T <: AbstractFloat}
    convert.(T, rand(icnf.rng, cstm, dims...))
end

@views function inference_sol(
    icnf::AbstractFlows{T, <:VectorMode},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractVector{<:Real}, NTuple{2, T}, false},
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    sol = solve(prob; icnf.sol_kwargs...)
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

@views function inference_sol(
    icnf::AbstractFlows{T, <:MatrixMode},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractMatrix{<:Real}, NTuple{2, T}, false},
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    sol = solve(prob; icnf.sol_kwargs...)
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

@views function generate_sol(
    icnf::AbstractFlows{T, <:VectorMode},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractVector{<:Real}, NTuple{2, T}, false},
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    sol = solve(prob; icnf.sol_kwargs...)
    fsol = sol[:, end]
    z = fsol[begin:(end - n_aug_input - n_aug - 1)]
    z
end

@views function generate_sol(
    icnf::AbstractFlows{T, <:MatrixMode},
    mode::Mode,
    prob::SciMLBase.AbstractODEProblem{<:AbstractMatrix{<:Real}, NTuple{2, T}, false},
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    sol = solve(prob; icnf.sol_kwargs...)
    fsol = sol[:, :, end]
    z = fsol[begin:(end - n_aug_input - n_aug - 1), :]
    z
end

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
    sol_args::Tuple = (),
    sol_kwargs::Dict = Dict(
        :alg_hints => [:nonstiff, :memorybound],
        :reltol => 1e-2 + eps(1e-2),
    ),
)
    steerdist = Uniform{data_type}(-steer_rate, steer_rate)
    _fnn(x, ps, st) = first(nn(x, ps, st))

    aicnf{
        data_type,
        compute_mode,
        !iszero(naugmented),
        !iszero(steer_rate),
        typeof(nn),
        typeof(resource),
        typeof(basedist),
        typeof(tspan),
        typeof(steerdist),
        typeof(differentiation_backend),
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
        sol_args,
        sol_kwargs,
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
    icnf::AbstractFlows{T, <:ComputeMode, AUGMENTED, true},
    ::TrainMode;
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    rng::AbstractRNG = Random.default_rng(),
) where {T <: AbstractFloat, AUGMENTED}
    t₀, t₁ = tspan
    Δt = abs(t₁ - t₀)
    r = convert(T, rand(rng, steerdist))
    t₁_new = muladd(Δt, r, t₁)
    (t₀, t₁_new)
end

@inline function steer_tspan(
    icnf::AbstractFlows,
    ::Mode;
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    rng::AbstractRNG = Random.default_rng(),
)
    tspan
end

@inline function zeros_T_AT(
    resource::AbstractResource,
    ::AbstractFlows{T},
    dims...,
) where {T <: AbstractFloat}
    zeros(T, dims...)
end

@inline function rand_T_AT(
    resource::AbstractResource,
    ::AbstractFlows{T},
    rng::AbstractRNG = Random.default_rng(),
    dims...,
) where {T <: AbstractFloat}
    rand(rng, T, dims...)
end

@inline function randn_T_AT(
    resource::AbstractResource,
    ::AbstractFlows{T},
    rng::AbstractRNG = Random.default_rng(),
    dims...,
) where {T <: AbstractFloat}
    randn(rng, T, dims...)
end

@inline function rand_cstm_AT(
    resource::AbstractResource,
    ::AbstractFlows{T},
    cstm::Any,
    rng::AbstractRNG = Random.default_rng(),
    dims...,
) where {T <: AbstractFloat}
    convert.(T, rand(rng, cstm, dims...))
end

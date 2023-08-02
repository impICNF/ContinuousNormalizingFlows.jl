export construct

function construct(
    aicnf::Type{<:AbstractFlows},
    nn,
    nvars::Integer,
    naugmented::Integer = 0;
    data_type::Type{<:AbstractFloat} = Float32,
    array_type::Type{<:AbstractArray} = Array,
    compute_mode::Type{<:ComputeMode} = ADVectorMode,
    augmented::Bool = false,
    steer::Bool = false,
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
    !augmented && !iszero(naugmented) && error("'naugmented' > 0: 'augmented' must be true")
    !steer && !iszero(steer_rate) && error("'steer_rate' > 0: 'steer' must be true")

    aicnf{data_type, array_type, compute_mode, augmented, steer}(
        nn,
        nvars,
        naugmented,
        basedist,
        tspan,
        steer_rate,
        differentiation_backend,
        sol_args,
        sol_kwargs,
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

@inline function n_augment_input(
    icnf::AbstractFlows{<:AbstractFloat, <:AbstractArray, <:ComputeMode, true},
)
    icnf.naugmented
end

@inline function n_augment_input(::AbstractFlows)
    0
end

@inline function steer_rate_value(
    icnf::AbstractFlows{<:AbstractFloat, <:AbstractArray, <:ComputeMode, AUGMENTED, true},
) where {AUGMENTED}
    icnf.steer_rate
end

@inline function steer_rate_value(::AbstractFlows{T}) where {T <: AbstractFloat}
    zero(T)
end

@inline function steer_tspan(
    icnf::AbstractFlows{T, <:AbstractArray, <:ComputeMode, AUGMENTED, true},
    ::TrainMode,
    tspan::NTuple{2} = icnf.tspan,
    steer_rate::AbstractFloat = steer_rate_value(icnf),
    rng::AbstractRNG = Random.default_rng(),
) where {T <: AbstractFloat, AUGMENTED}
    t₀, t₁ = tspan
    steer_b = steer_rate * t₁
    d_s = Uniform{T}(t₁ - steer_b, t₁ + steer_b)
    t₁_new = convert(T, rand(rng, d_s))
    (t₀, t₁_new)
end

@inline function steer_tspan(
    icnf::AbstractFlows,
    ::Mode,
    tspan::NTuple{2} = icnf.tspan,
    steer_rate::AbstractFloat = steer_rate_value(icnf),
    rng::AbstractRNG = Random.default_rng(),
)
    tspan
end

@inline function zeros_T_AT(
    ::AbstractFlows{T, <:CuArray},
    dims...,
) where {T <: AbstractFloat}
    CUDA.zeros(T, dims...)
end

@inline function zeros_T_AT(::AbstractFlows{T}, dims...) where {T <: AbstractFloat}
    zeros(T, dims...)
end

@inline function rand_T_AT(
    ::AbstractFlows{T, <:CuArray},
    rng::AbstractRNG = Random.default_rng(),
    dims...,
) where {T <: AbstractFloat}
    CUDA.rand(T, dims...)
end

@inline function rand_T_AT(
    ::AbstractFlows{T},
    rng::AbstractRNG = Random.default_rng(),
    dims...,
) where {T <: AbstractFloat}
    rand(rng, T, dims...)
end

@inline function randn_T_AT(
    ::AbstractFlows{T, <:CuArray},
    rng::AbstractRNG = Random.default_rng(),
    dims...,
) where {T <: AbstractFloat}
    CUDA.randn(T, dims...)
end

@inline function randn_T_AT(
    ::AbstractFlows{T},
    rng::AbstractRNG = Random.default_rng(),
    dims...,
) where {T <: AbstractFloat}
    randn(rng, T, dims...)
end

@non_differentiable CUDA.zeros(::Any...)
@non_differentiable CUDA.rand(::Any...)
@non_differentiable CUDA.randn(::Any...)

export construct

function construct(
    aicnf::Type{<:AbstractFlows},
    nn,
    nvars::Integer;
    data_type::Type{<:AbstractFloat} = Float32,
    array_type::Type{<:AbstractArray} = Array,
    compute_mode::Type{<:ComputeMode} = ADVectorMode,
    basedist::Distribution = MvNormal(Zeros{data_type}(nvars), one(data_type) * I),
    tspan::NTuple{2} = (zero(data_type), one(data_type)),
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    sol_args::Tuple = (),
    sol_kwargs::Dict = Dict(
        :alg_hints => [:nonstiff, :memorybound],
        :reltol => 1e-2 + eps(1e-2),
    ),
)
    aicnf{data_type, array_type, compute_mode}(
        nn,
        nvars,
        basedist,
        tspan,
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
        "\n\tTime Span: ",
        icnf.tspan,
    )
end

@inline function zeros_T_AT(
    ::AbstractFlows{T, AT},
    dims...,
) where {T <: AbstractFloat, AT <: AbstractArray}
    if AT <: CuArray
        CUDA.zeros(T, dims...)
    else
        zeros(T, dims...)
    end
end

@inline function rand_T_AT(
    ::AbstractFlows{T, AT},
    rng::AbstractRNG = Random.default_rng(),
    dims...,
) where {T <: AbstractFloat, AT <: AbstractArray}
    if AT <: CuArray
        CUDA.rand(T, dims...)
    else
        rand(rng, T, dims...)
    end
end

@inline function randn_T_AT(
    ::AbstractFlows{T, AT},
    rng::AbstractRNG = Random.default_rng(),
    dims...,
) where {T <: AbstractFloat, AT <: AbstractArray}
    if AT <: CuArray
        CUDA.randn(T, dims...)
    else
        randn(rng, T, dims...)
    end
end

@non_differentiable CUDA.zeros(::Any...)
@non_differentiable CUDA.rand(::Any...)
@non_differentiable CUDA.randn(::Any...)

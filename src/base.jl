export construct

function construct(
    aicnf::Type{<:AbstractFlows},
    nn,
    nvars::Integer;
    data_type::Type{<:AbstractFloat} = Float32,
    array_type::Type{<:AbstractArray} = Array,
    compute_mode::Type{<:ComputeMode} = ADVectorMode,
    basedist::Distribution = MvNormal(Zeros{data_type}(nvars), one(data_type) * I),
    tspan::Tuple = (zero(data_type), one(data_type)),
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

function n_augment(icnf::AbstractFlows, mode::Mode)::Integer
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

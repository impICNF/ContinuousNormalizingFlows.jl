export construct

function construct(
    aicnf::Type{<:AbstractFlows},
    nn,
    nvars::Integer,
    ;
    data_type::Type{<:AbstractFloat} = Float32,
    array_type::Type{<:AbstractArray} = Array,
    basedist::Distribution = MvNormal(Zeros{data_type}(nvars), one(data_type) * I),
    tspan::Tuple = convert(Tuple{data_type, data_type}, (0, 1)),
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
)
    aicnf{data_type, array_type}(nn, nvars, basedist, tspan, differentiation_backend)
end

function n_augment(
    icnf::AbstractFlows,
    mode::Mode,
)::Integer
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

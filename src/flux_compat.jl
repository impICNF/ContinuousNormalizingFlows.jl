export FluxCompatLayer

struct FluxCompatLayer{L, RE, I} <: LuxCore.AbstractExplicitLayer
    layer::L
    re::RE
    init_parameters::I
end

function FluxCompatLayer(l)
    p, re = Optimisers.destructure(l)
    p_ = copy(p)
    return FluxCompatLayer(l, re, () -> p_)
end

Lux.initialparameters(::AbstractRNG, l::FluxCompatLayer) = l.init_parameters()

(l::FluxCompatLayer)(x, ps, st) = l.re(ps)(x), st

Base.show(io::IO, l::FluxCompatLayer) = print(io, "FluxCompatLayer($(l.layer))")

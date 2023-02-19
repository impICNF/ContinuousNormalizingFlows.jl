export MyFluxLayer

struct MyFluxLayer{L, RE, I} <: LuxCore.AbstractExplicitLayer
    layer::L
    re::RE
    init_parameters::I
end

function MyFluxLayer(l)
    p, re = Optimisers.destructure(l)
    p_ = copy(p)
    return MyFluxLayer(l, re, () -> p_)
end

Lux.initialparameters(::AbstractRNG, l::MyFluxLayer) = l.init_parameters()

(l::MyFluxLayer)(x, ps, st) = l.re(ps)(x), st

Base.show(io::IO, l::MyFluxLayer) = print(io, "MyFluxLayer($(l.layer))")

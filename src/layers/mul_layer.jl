struct MulLayer{F1, F2, NVARS <: Int} <: LuxCore.AbstractExplicitLayer
    activation::F1
    nvars::NVARS
    init_weight::F2
end

function MulLayer(
    nvars::Int,
    activation::Any = identity;
    init_weight::Any = Lux.glorot_uniform,
    allow_fast_activation::Bool = true,
)
    activation = ifelse(allow_fast_activation, NNlib.fast_act(activation), activation)
    MulLayer{typeof(activation), typeof(init_weight), typeof(nvars)}(
        activation,
        nvars,
        init_weight,
    )
end

function LuxCore.initialparameters(rng::AbstractRNG, m::MulLayer)
    (weight = m.init_weight(rng, m.nvars, m.nvars),)
end

function LuxCore.parameterlength(m::MulLayer)
    m.nvars * m.nvars
end

function LuxCore.outputsize(m::MulLayer)
    (m.nvars,)
end

@inline function (m::MulLayer)(x::AbstractVecOrMat, ps::Any, st::NamedTuple)
    return Lux.__apply_activation(m.activation, Octavian.matmul(ps.weight, x)), st
end

"""
Implementation of Planar Layer from

[Chen, Ricky TQ, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. "Neural Ordinary Differential Equations." arXiv preprint arXiv:1806.07366 (2018).](https://arxiv.org/abs/1806.07366)
"""
struct PlanarLayer{USE_BIAS, F1, F2, F3, NVARIABLES <: Int} <: LuxCore.AbstractLuxLayer
    in_dims::NVARIABLES
    out_dims::NVARIABLES
    activation::F1
    init_weight::F2
    init_bias::F3
end

function PlanarLayer(
    mapping::Pair{<:Int, <:Int},
    activation::Any = identity;
    init_weight::Any = WeightInitializers.glorot_uniform,
    init_bias::Any = WeightInitializers.zeros32,
    use_bias::Bool = true,
)
    return PlanarLayer{
        use_bias,
        typeof(activation),
        typeof(init_weight),
        typeof(init_bias),
        typeof(first(mapping)),
    }(
        first(mapping),
        last(mapping),
        activation,
        init_weight,
        init_bias,
    )
end

function LuxCore.initialparameters(rng::Random.AbstractRNG, layer::PlanarLayer{true})
    return (
        u = layer.init_weight(rng, layer.out_dims),
        w = layer.init_weight(rng, layer.in_dims),
        b = layer.init_bias(rng, 1),
    )
end

function LuxCore.initialparameters(rng::Random.AbstractRNG, layer::PlanarLayer{false})
    return (
        u = layer.init_weight(rng, layer.out_dims),
        w = layer.init_weight(rng, layer.in_dims),
    )
end

function LuxCore.parameterlength(m::PlanarLayer{USE_BIAS}) where {USE_BIAS}
    return m.out_dims + m.in_dims + USE_BIAS
end

function LuxCore.outputsize(m::PlanarLayer, ::Any, ::Random.AbstractRNG)
    return (m.out_dims,)
end

function (m::PlanarLayer{true})(z::AbstractVector{<:Real}, ps::Any, st::NamedTuple)
    activation = NNlib.fast_act(m.activation, z)
    return ps.u * activation.(LinearAlgebra.dot(ps.w, z) + only(ps.b)), st
end

function (m::PlanarLayer{true})(z::AbstractMatrix{<:Real}, ps::Any, st::NamedTuple)
    activation = NNlib.fast_act(m.activation, z)
    return ps.u * activation.(muladd(permutedims(ps.w), z, only(ps.b))), st
end

function (m::PlanarLayer{false})(z::AbstractVector{<:Real}, ps::Any, st::NamedTuple)
    activation = NNlib.fast_act(m.activation, z)
    return ps.u * activation.(LinearAlgebra.dot(ps.w, z)), st
end

function (m::PlanarLayer{false})(z::AbstractMatrix{<:Real}, ps::Any, st::NamedTuple)
    activation = NNlib.fast_act(m.activation, z)
    return ps.u * activation.(permutedims(ps.w) * z), st
end

function pl_h(m::PlanarLayer{true}, z::AbstractVector{<:Real}, ps::Any, st::NamedTuple)
    activation = NNlib.fast_act(m.activation, z)
    return activation.(LinearAlgebra.dot(ps.w, z) + only(ps.b)), st
end

function pl_h(m::PlanarLayer{true}, z::AbstractMatrix{<:Real}, ps::Any, st::NamedTuple)
    activation = NNlib.fast_act(m.activation, z)
    return activation.(muladd(permutedims(ps.w), z, only(ps.b))), st
end

function pl_h(m::PlanarLayer{false}, z::AbstractVector{<:Real}, ps::Any, st::NamedTuple)
    activation = NNlib.fast_act(m.activation, z)
    return activation.(LinearAlgebra.dot(ps.w, z)), st
end

function pl_h(m::PlanarLayer{false}, z::AbstractMatrix{<:Real}, ps::Any, st::NamedTuple)
    activation = NNlib.fast_act(m.activation, z)
    return activation.(permutedims(ps.w) * z), st
end

"""
Implementation of Planar Layer from

[Chen, Ricky TQ, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. "Neural Ordinary Differential Equations." arXiv preprint arXiv:1806.07366 (2018).](https://arxiv.org/abs/1806.07366)
"""
struct PlanarLayer{use_bias, cond, F1, F2, F3, NVARS <: Int} <: LuxCore.AbstractLuxLayer
    activation::F1
    nvars::NVARS
    init_weight::F2
    init_bias::F3
    n_cond::NVARS
end

function PlanarLayer(
    nvars::Int,
    activation::Any = identity;
    init_weight::Any = Lux.glorot_uniform,
    init_bias::Any = Lux.zeros32,
    use_bias::Bool = true,
    n_cond::Int = 0,
)
    PlanarLayer{
        use_bias,
        !iszero(n_cond),
        typeof(activation),
        typeof(init_weight),
        typeof(init_bias),
        typeof(nvars),
    }(
        activation,
        nvars,
        init_weight,
        init_bias,
        n_cond,
    )
end

function LuxCore.initialparameters(
    rng::Random.AbstractRNG,
    layer::PlanarLayer{use_bias, cond},
) where {use_bias, cond}
    ifelse(
        use_bias,
        (
            u = layer.init_weight(rng, layer.nvars),
            w = layer.init_weight(
                rng,
                ifelse(cond, (layer.nvars + layer.n_cond), layer.nvars),
            ),
            b = layer.init_bias(rng, 1),
        ),
        (
            u = layer.init_weight(rng, layer.nvars),
            w = layer.init_weight(
                rng,
                ifelse(cond, (layer.nvars + layer.n_cond), layer.nvars),
            ),
        ),
    )
end

function LuxCore.parameterlength(m::PlanarLayer{use_bias, cond}) where {use_bias, cond}
    m.nvars + ifelse(cond, (m.nvars + m.n_cond), m.nvars) + ifelse(use_bias, 1, 0)
end

function LuxCore.outputsize(m::PlanarLayer, ::Any, ::AbstractRNG)
    (m.nvars,)
end

@inline function (m::PlanarLayer{true})(z::AbstractVector, ps::Any, st::NamedTuple)
    activation = NNlib.fast_act(m.activation, z)
    ps.u * activation.(LinearAlgebra.dot(ps.w, z) + only(ps.b)), st
end

@inline function (m::PlanarLayer{true})(z::AbstractMatrix, ps::Any, st::NamedTuple)
    activation = NNlib.fast_act(m.activation, z)
    ps.u * activation.(muladd(transpose(ps.w), z, only(ps.b))), st
end

@inline function (m::PlanarLayer{false})(z::AbstractVector, ps::Any, st::NamedTuple)
    activation = NNlib.fast_act(m.activation, z)
    ps.u * activation.(LinearAlgebra.dot(ps.w, z)), st
end

@inline function (m::PlanarLayer{false})(z::AbstractMatrix, ps::Any, st::NamedTuple)
    activation = NNlib.fast_act(m.activation, z)
    ps.u * activation.(transpose(ps.w) * z), st
end

@inline function pl_h(m::PlanarLayer{true}, z::AbstractVector, ps::Any, st::NamedTuple)
    activation = NNlib.fast_act(m.activation, z)
    activation.(LinearAlgebra.dot(ps.w, z) + only(ps.b)), st
end

@inline function pl_h(m::PlanarLayer{true}, z::AbstractMatrix, ps::Any, st::NamedTuple)
    activation = NNlib.fast_act(m.activation, z)
    activation.(muladd(transpose(ps.w), z, only(ps.b))), st
end

@inline function pl_h(m::PlanarLayer{false}, z::AbstractVector, ps::Any, st::NamedTuple)
    activation = NNlib.fast_act(m.activation, z)
    activation.(LinearAlgebra.dot(ps.w, z)), st
end

@inline function pl_h(m::PlanarLayer{false}, z::AbstractMatrix, ps::Any, st::NamedTuple)
    activation = NNlib.fast_act(m.activation, z)
    activation.(transpose(ps.w) * z), st
end

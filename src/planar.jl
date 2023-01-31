export Planar, PlanarNN

struct PlanarNN <: LuxCore.AbstractExplicitLayer
    nvars::Integer
    h::Function
    cond::Bool
end

function PlanarNN(
    nvars::Integer,
    h::Function = tanh;
    cond::Bool = false,
)
    PlanarNN(nvars, NNlib.fast_act(h), cond)
end

function LuxCore.initialparameters(rng::AbstractRNG, layer::PlanarNN)
    (
        u = randn(rng, layer.nvars),
        w = randn(rng, layer.cond ? layer.nvars * 2 : layer.nvars),
        b = randn(rng, 1),
    )
end

function (m::PlanarNN)(
    z::AbstractVector{<:Real},
    ps::Any,
    st::Any,
)::Tuple{<:AbstractVecOrMat, <:NamedTuple}
    ps.u * m.h((ps.w ⋅ z) + only(ps.b)), st
end

function pl_h(
    m::PlanarNN,
    z::AbstractVector{<:Real},
    ps::Any,
    st::Any,
)::Tuple{<:Union{AbstractVecOrMat, Real}, <:NamedTuple}
    m.h((ps.w ⋅ z) + only(ps.b)), st
end

"""
Implementation of Planar Flows from

[Chen, Ricky TQ, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. "Neural Ordinary Differential Equations." arXiv preprint arXiv:1806.07366 (2018).](https://arxiv.org/abs/1806.07366)
"""
struct Planar{T <: AbstractFloat, AT <: AbstractArray} <: AbstractICNF{T, AT}
    nn::PlanarNN

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    # trace_test
    # trace_train
end

function Planar{T, AT}(
    nn::PlanarNN,
    nvars::Integer,
    ;
    basedist::Distribution = MvNormal(Zeros{T}(nvars), one(T) * I),
    tspan::Tuple{T, T} = convert(Tuple{T, T}, default_tspan),
) where {T <: AbstractFloat, AT <: AbstractArray}
    Planar{T, AT}(nn, nvars, basedist, tspan)
end

function augmented_f(
    icnf::Planar{T, AT},
    mode::Mode,
    st::Any;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1

    function f_aug(u, p, t)
        z = u[1:(end - n_aug)]
        mz, _ = LuxCore.apply(icnf.nn, z, p, st)
        trace_J =
            p.u ⋅ transpose(
                only(
                    AbstractDifferentiation.jacobian(
                        differentiation_backend,
                        x -> first(pl_h(icnf.nn, x, p, st)),
                        z,
                    ),
                ),
            )
        vcat(mz, -trace_J)
    end
    f_aug
end

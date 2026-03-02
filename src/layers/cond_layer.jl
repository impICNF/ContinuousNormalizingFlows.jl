struct CondLayer{NN <: LuxCore.AbstractLuxLayer, AT <: Any} <:
       LuxCore.AbstractLuxWrapperLayer{:nn}
    nn::NN
    ys::AT
end

function (m::CondLayer{<:LuxCore.AbstractLuxLayer, <:AbstractVecOrMat{<:Real}})(
    z::AbstractVecOrMat{<:Real},
    ps::Any,
    st::NamedTuple,
)
    return LuxCore.apply(m.nn, vcat(z, m.ys), ps, st)
end

function (m::CondLayer{<:LuxCore.AbstractLuxLayer, <:Number})(
    z::AbstractVector{<:Real},
    ps::Any,
    st::NamedTuple,
)
    return LuxCore.apply(m.nn, vcat(z, m.ys), ps, st)
end

function (m::CondLayer{<:LuxCore.AbstractLuxLayer, <:Number})(
    z::AbstractMatrix{<:Real},
    ps::Any,
    st::NamedTuple,
)
    ts = similar(z, 1, size(z, 2))
    ChainRulesCore.@ignore_derivatives fill!(ts, m.ys)
    return LuxCore.apply(m.nn, vcat(z, ts), ps, st)
end

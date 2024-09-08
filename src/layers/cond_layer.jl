struct CondLayer{NN <: LuxCore.AbstractLuxLayer, AT <: AbstractArray} <:
       LuxCore.AbstractLuxContainerLayer{(:nn,)}
    nn::NN
    ys::AT
end

@inline function (m::CondLayer)(z::AbstractVecOrMat, ps::Any, st::NamedTuple)
    LuxCore.apply(m.nn, vcat(z, m.ys), ps, st)
end

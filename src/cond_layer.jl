struct CondLayer{NN <: LuxCore.AbstractExplicitLayer, AT <: AbstractArray} <:
       LuxCore.AbstractExplicitContainerLayer{(:nn,)}
    nn::NN
    ys::AT
end

@inline function (m::CondLayer)(z::AbstractArray, ps::Any, st::Any)
    m.nn(vcat(z, m.ys), ps, st)
end

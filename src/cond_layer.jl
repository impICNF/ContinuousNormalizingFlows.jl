struct CondLayer{NN <: LuxCore.AbstractExplicitLayer, AT <: AbstractArray} <:
       LuxCore.AbstractExplicitLayer
    nn::NN
    ys::AT
end

function LuxCore.initialparameters(rng::AbstractRNG, layer::CondLayer)
    LuxCore.initialparameters(rng, layer.nn)
end

@inline function (m::CondLayer)(z::AbstractArray, ps::Any, st::Any)
    m.nn(vcat(z, m.ys), ps, st)
end

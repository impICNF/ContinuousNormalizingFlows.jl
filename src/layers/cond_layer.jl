struct CondLayer{NN <: LuxCore.AbstractLuxLayer, AT <: Any} <:
       LuxCore.AbstractLuxWrapperLayer{:nn}
    nn::NN
    ys::AT
end

function (m::CondLayer)(z::AbstractVecOrMat, ps::Any, st::NamedTuple)
    return LuxCore.apply(m.nn, vcat(z, m.ys), ps, st)
end

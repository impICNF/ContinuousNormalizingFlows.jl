function icnf_jacobian(
    icnf::AbstractICNF{<:AbstractFloat, <:DIVectorMode},
    ::TestMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractVector{<:Real},
)
    y, J = DifferentiationInterface.value_and_jacobian(f, icnf.compute_mode.adback, xs)
    return y, oftype(y, J)
end

function icnf_jacobian(
    icnf::AbstractICNF{<:AbstractFloat, <:DIMatrixMode},
    ::TestMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractMatrix{<:Real},
)
    y, J = DifferentiationInterface.value_and_jacobian(f, icnf.compute_mode.adback, xs)
    return y, oftype.(Ref(y), split_jac(J, size(xs, 1)))
end

function icnf_jacobian(
    icnf::AbstractICNF{T, <:DIVecJacMatrixMode},
    ::TestMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractMatrix{<:Real},
) where {T}
    y = f(xs)
    z = similar(xs)
    ChainRulesCore.@ignore_derivatives fill!(z, zero(T))
    res = Zygote.Buffer(y, size(xs, 1), size(xs, 1), size(xs, 2))
    for i in axes(xs, 1)
        ChainRulesCore.@ignore_derivatives z[i, :] .= one(T)
        res[i, :, :] =
            only(DifferentiationInterface.pullback(f, icnf.compute_mode.adback, xs, (z,)))
        ChainRulesCore.@ignore_derivatives z[i, :] .= zero(T)
    end
    return y, oftype.(Ref(y), eachslice(copy(res); dims = 3))
end

function icnf_jacobian(
    icnf::AbstractICNF{T, <:DIJacVecMatrixMode},
    ::TestMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractMatrix{<:Real},
) where {T}
    y = f(xs)
    z = similar(xs)
    ChainRulesCore.@ignore_derivatives fill!(z, zero(T))
    res = Zygote.Buffer(y, size(xs, 1), size(xs, 1), size(xs, 2))
    for i in axes(xs, 1)
        ChainRulesCore.@ignore_derivatives z[i, :] .= one(T)
        res[:, i, :] = only(
            DifferentiationInterface.pushforward(f, icnf.compute_mode.adback, xs, (z,)),
        )
        ChainRulesCore.@ignore_derivatives z[i, :] .= zero(T)
    end
    return y, oftype.(Ref(y), eachslice(copy(res); dims = 3))
end

function icnf_jacobian(
    icnf::AbstractICNF{<:AbstractFloat, <:LuxMatrixMode},
    ::TestMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractMatrix{<:Real},
)
    y = f(xs)
    return y,
    oftype.(
        Ref(y),
        eachslice(Lux.batched_jacobian(f, icnf.compute_mode.adback, xs); dims = 3),
    )
end

function icnf_jacobian(
    icnf::AbstractICNF{T, <:DIVecJacVectorMode},
    ::TrainMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat}
    y, ϵJ =
        DifferentiationInterface.value_and_pullback(f, icnf.compute_mode.adback, xs, (ϵ,))
    return y, oftype(y, only(ϵJ))
end

function icnf_jacobian(
    icnf::AbstractICNF{T, <:DIJacVecVectorMode},
    ::TrainMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat}
    y, Jϵ = DifferentiationInterface.value_and_pushforward(
        f,
        icnf.compute_mode.adback,
        xs,
        (ϵ,),
    )
    return y, oftype(y, only(Jϵ))
end

function icnf_jacobian(
    icnf::AbstractICNF{T, <:DIVecJacMatrixMode},
    ::TrainMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat}
    y, ϵJ =
        DifferentiationInterface.value_and_pullback(f, icnf.compute_mode.adback, xs, (ϵ,))
    return y, oftype(y, only(ϵJ))
end

function icnf_jacobian(
    icnf::AbstractICNF{T, <:DIJacVecMatrixMode},
    ::TrainMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat}
    y, Jϵ = DifferentiationInterface.value_and_pushforward(
        f,
        icnf.compute_mode.adback,
        xs,
        (ϵ,),
    )
    return y, oftype(y, only(Jϵ))
end

function icnf_jacobian(
    icnf::AbstractICNF{T, <:LuxVecJacMatrixMode},
    ::TrainMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat}
    y = f(xs)
    return y, oftype(y, Lux.vector_jacobian_product(f, icnf.compute_mode.adback, xs, ϵ))
end

function icnf_jacobian(
    icnf::AbstractICNF{T, <:LuxJacVecMatrixMode},
    ::TrainMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat}
    y = f(xs)
    return y, oftype(y, Lux.jacobian_vector_product(f, icnf.compute_mode.adback, xs, ϵ))
end

function split_jac(x::AbstractMatrix{<:Real}, sz::Integer)
    return (
        x[i:j, i:j] for (i, j) in zip(
            firstindex(x, 1):sz:lastindex(x, 1),
            (firstindex(x, 1) + sz - 1):sz:lastindex(x, 1),
        )
    )
end

function icnf_jacobian(
    icnf::AbstractICNF{<:AbstractFloat, <:DIVectorMode},
    ::TestMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractVector{<:Real},
)
    y = f(xs)
    return y,
    oftype(hcat(y), DifferentiationInterface.jacobian(f, icnf.compute_mode.adback, xs))
end

function icnf_jacobian(
    icnf::AbstractICNF{<:AbstractFloat, <:DIMatrixMode},
    ::TestMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractMatrix{<:Real},
)
    y = f(xs)
    J = DifferentiationInterface.jacobian(f, icnf.compute_mode.adback, xs)
    return y,
    oftype(
        cat(y; dims = Val(3)),
        cat(
            (
                J[i:j, i:j] for (i, j) in zip(
                    firstindex(J, 1):size(y, 1):lastindex(J, 1),
                    (firstindex(J, 1) + size(y, 1) - 1):size(y, 1):lastindex(J, 1),
                )
            )...;
            dims = Val(3),
        ),
    )
end

function icnf_jacobian(
    icnf::AbstractICNF{T, <:DIVecJacMatrixMode},
    ::TestMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractMatrix{<:Real},
) where {T <: AbstractFloat}
    y = f(xs)
    z = similar(xs)
    ChainRulesCore.@ignore_derivatives fill!(z, zero(T))
    res = Zygote.Buffer(y, size(xs, 1), size(xs, 1), size(xs, 2))
    for i in axes(xs, 1)
        ChainRulesCore.@ignore_derivatives z[i, :] .= one(T)
        res[i, :, :] = oftype(
            y,
            only(DifferentiationInterface.pullback(f, icnf.compute_mode.adback, xs, (z,))),
        )
        ChainRulesCore.@ignore_derivatives z[i, :] .= zero(T)
    end
    return y, oftype(cat(y; dims = Val(3)), copy(res))
end

function icnf_jacobian(
    icnf::AbstractICNF{T, <:DIJacVecMatrixMode},
    ::TestMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractMatrix{<:Real},
) where {T <: AbstractFloat}
    y = f(xs)
    z = similar(xs)
    ChainRulesCore.@ignore_derivatives fill!(z, zero(T))
    res = Zygote.Buffer(y, size(xs, 1), size(xs, 1), size(xs, 2))
    for i in axes(xs, 1)
        ChainRulesCore.@ignore_derivatives z[i, :] .= one(T)
        res[:, i, :] = oftype(
            y,
            only(
                DifferentiationInterface.pushforward(f, icnf.compute_mode.adback, xs, (z,)),
            ),
        )
        ChainRulesCore.@ignore_derivatives z[i, :] .= zero(T)
    end
    return y, oftype(cat(y; dims = Val(3)), copy(res))
end

function icnf_jacobian(
    icnf::AbstractICNF{<:AbstractFloat, <:LuxMatrixMode},
    ::TestMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractMatrix{<:Real},
)
    y = f(xs)
    return y,
    oftype(cat(y; dims = Val(3)), Lux.batched_jacobian(f, icnf.compute_mode.adback, xs))
end

function icnf_jacobian(
    icnf::AbstractICNF{T, <:DIVecJacVectorMode},
    ::TrainMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat}
    y = f(xs)
    return y,
    oftype(
        y,
        only(DifferentiationInterface.pullback(f, icnf.compute_mode.adback, xs, (ϵ,))),
    )
end

function icnf_jacobian(
    icnf::AbstractICNF{T, <:DIJacVecVectorMode},
    ::TrainMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat}
    y = f(xs)
    return y,
    oftype(
        y,
        only(DifferentiationInterface.pushforward(f, icnf.compute_mode.adback, xs, (ϵ,))),
    )
end

function icnf_jacobian(
    icnf::AbstractICNF{T, <:DIVecJacMatrixMode},
    ::TrainMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat}
    y = f(xs)
    return y,
    oftype(
        y,
        only(DifferentiationInterface.pullback(f, icnf.compute_mode.adback, xs, (ϵ,))),
    )
end

function icnf_jacobian(
    icnf::AbstractICNF{T, <:DIJacVecMatrixMode},
    ::TrainMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat}
    y = f(xs)
    return y,
    oftype(
        y,
        only(DifferentiationInterface.pushforward(f, icnf.compute_mode.adback, xs, (ϵ,))),
    )
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

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
        stack(
            function (i, j)
                return J[i:j, i:j]
            end,
            firstindex(J, 1):size(y, 1):lastindex(J, 1),
            (firstindex(J, 1) + size(y, 1) - 1):size(y, 1):lastindex(J, 1);
            dims = 3,
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
    ons = similar(xs, 1, size(xs, 2))
    ChainRulesCore.@ignore_derivatives fill!(ons, one(T))
    return y,
    oftype(
        cat(y; dims = Val(3)),
        stack(
            DifferentiationInterface.pullback(
                f,
                icnf.compute_mode.adback,
                xs,
                ntuple(function (i::Int)
                    return oftype(xs, (axes(xs, 1) .== i) * ons)
                end, size(xs, 1)),
            );
            dims = 1,
        ),
    )
end

function icnf_jacobian(
    icnf::AbstractICNF{T, <:DIJacVecMatrixMode},
    ::TestMode,
    f::LuxCore.StatefulLuxLayer,
    xs::AbstractMatrix{<:Real},
) where {T <: AbstractFloat}
    y = f(xs)
    ons = similar(xs, 1, size(xs, 2))
    ChainRulesCore.@ignore_derivatives fill!(ons, one(T))
    return y,
    oftype(
        cat(y; dims = Val(3)),
        stack(
            DifferentiationInterface.pushforward(
                f,
                icnf.compute_mode.adback,
                xs,
                ntuple(function (i::Int)
                    return oftype(xs, (axes(xs, 1) .== i) * ons)
                end, size(xs, 1)),
            );
            dims = 2,
        ),
    )
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

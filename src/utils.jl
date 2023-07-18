function jacobian_batched(
    icnf::AbstractFlows{T, AT, <:ZygoteMatrixMode},
    f,
    xs::AbstractMatrix{<:Real},
)::Tuple where {T <: AbstractFloat, AT <: AbstractArray}
    y, back = Zygote.pullback(f, xs)
    z = zeros_T_AT(icnf, size(xs))
    res = Zygote.Buffer(xs, size(xs, 1), size(xs, 1), size(xs, 2))
    for i in axes(xs, 1)
        @ignore_derivatives z[i, :] .= one(T)
        res[i, :, :] = only(back(z))
        @ignore_derivatives z[i, :] .= zero(T)
    end
    y, copy(res)
end

function jacobian_batched(
    icnf::AbstractFlows{T, AT, <:SDVecJacMatrixMode},
    f,
    xs::AbstractMatrix{<:Real},
)::Tuple where {T <: AbstractFloat, AT <: AbstractArray}
    y = f(xs)
    z = zeros_T_AT(icnf, size(xs))
    res = Zygote.Buffer(xs, size(xs, 1), size(xs, 1), size(xs, 2))
    for i in axes(xs, 1)
        @ignore_derivatives z[i, :] .= one(T)
        res[i, :, :] = reshape(auto_vecjac(f, xs, z), size(xs))
        @ignore_derivatives z[i, :] .= zero(T)
    end
    y, copy(res)
end

function jacobian_batched(
    icnf::AbstractFlows{T, AT, <:SDJacVecMatrixMode},
    f,
    xs::AbstractMatrix{<:Real},
)::Tuple where {T <: AbstractFloat, AT <: AbstractArray}
    y = f(xs)
    z = zeros_T_AT(icnf, size(xs))
    res = Zygote.Buffer(xs, size(xs, 1), size(xs, 1), size(xs, 2))
    for i in axes(xs, 1)
        @ignore_derivatives z[i, :] .= one(T)
        res[i, :, :] = reshape(auto_jacvec(f, xs, z), size(xs))
        @ignore_derivatives z[i, :] .= zero(T)
    end
    y, copy(res)
end

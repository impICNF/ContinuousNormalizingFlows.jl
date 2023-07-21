function jacobian_batched(
    icnf::AbstractFlows{<:AbstractFloat, <:AbstractArray, <:ZygoteMatrixMode},
    f,
    xs::AbstractMatrix{<:Real},
)
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
    icnf::AbstractFlows{<:AbstractFloat, <:AbstractArray, <:SDVecJacMatrixMode},
    f,
    xs::AbstractMatrix{<:Real},
)
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
    icnf::AbstractFlows{<:AbstractFloat, <:AbstractArray, <:SDJacVecMatrixMode},
    f,
    xs::AbstractMatrix{<:Real},
)
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

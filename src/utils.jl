function jacobian_batched(
    f,
    xs::AbstractMatrix{<:Real},
    T::Type{<:AbstractFloat},
    AT::Type{<:AbstractArray},
    CM::Type{<:ZygoteMatrixMode},
)::Tuple
    y, back = Zygote.pullback(f, xs)
    z::AT = zeros(T, size(xs))
    res = Zygote.Buffer(xs, size(xs, 1), size(xs, 1), size(xs, 2))
    for i in axes(xs, 1)
        z[i, :] .= one(T)
        res[i, :, :] = only(back(z))
        z[i, :] .= zero(T)
    end
    y, copy(res)
end

function jacobian_batched(
    f,
    xs::AbstractMatrix{<:Real},
    T::Type{<:AbstractFloat},
    AT::Type{<:AbstractArray},
    CM::Type{<:SDVecJacMatrixMode},
)::Tuple
    y = f(xs)
    z::AT = zeros(T, size(xs))
    res = Zygote.Buffer(xs, size(xs, 1), size(xs, 1), size(xs, 2))
    for i in axes(xs, 1)
        z[i, :] .= one(T)
        res[i, :, :] = reshape(auto_vecjac(f, xs, z), size(xs))
        z[i, :] .= zero(T)
    end
    y, copy(res)
end

function jacobian_batched(
    f,
    xs::AbstractMatrix{<:Real},
    T::Type{<:AbstractFloat},
    AT::Type{<:AbstractArray},
    CM::Type{<:SDJacVecMatrixMode},
)::Tuple
    y = f(xs)
    z::AT = zeros(T, size(xs))
    res = Zygote.Buffer(xs, size(xs, 1), size(xs, 1), size(xs, 2))
    for i in axes(xs, 1)
        z[i, :] .= one(T)
        res[i, :, :] = reshape(auto_jacvec(f, xs, z), size(xs))
        z[i, :] .= zero(T)
    end
    y, copy(res)
end

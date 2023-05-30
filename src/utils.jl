function jacobian_batched(
    f,
    xs::AbstractMatrix{<:AbstractFloat},
    T::Type{<:AbstractFloat},
    AT::Type{<:AbstractArray},
    CM::Type{<:ZygoteMatrixMode},
)::Tuple
    y, back = Zygote.pullback(f, xs)
    z = convert(AT, zeros(T, size(xs)))
    res = convert(AT, zeros(T, size(xs, 1), size(xs, 1), size(xs, 2)))
    for i in axes(y, 1)
        z[i, :] .= one(T)
        res[i, :, :] .= only(back(z))
        z[i, :] .= zero(T)
    end
    y, res
end

function jacobian_batched(
    f,
    xs::AbstractMatrix{<:AbstractFloat},
    T::Type{<:AbstractFloat},
    AT::Type{<:AbstractArray},
    CM::Type{<:SDVecJacMatrixMode},
)::Tuple
    y = f(xs)
    z = convert(AT, zeros(T, size(xs)))
    res = convert(AT, zeros(T, size(xs, 1), size(xs, 1), size(xs, 2)))
    for i in axes(y, 1)
        z[i, :] .= one(T)
        res[i, :, :] .= reshape(auto_vecjac(f, xs, z), size(xs))
        z[i, :] .= zero(T)
    end
    y, res
end

function jacobian_batched(
    f,
    xs::AbstractMatrix{<:AbstractFloat},
    T::Type{<:AbstractFloat},
    AT::Type{<:AbstractArray},
    CM::Type{<:SDJacVecMatrixMode},
)::Tuple
    y = f(xs)
    z = convert(AT, zeros(T, size(xs)))
    res = convert(AT, zeros(T, size(xs, 1), size(xs, 1), size(xs, 2)))
    for i in axes(y, 1)
        z[i, :] .= one(T)
        res[i, :, :] .= reshape(auto_jacvec(f, xs, z), size(xs))
        z[i, :] .= zero(T)
    end
    y, res
end

function jacobian_batched(
    f,
    xs::AbstractMatrix{<:Real},
    T::Type{<:AbstractFloat},
    AT::Type{<:AbstractArray},
    CM::Type{<:ZygoteMatrixMode},
)::Tuple
    y, back = Zygote.pullback(f, xs)
    z = convert(AT, zeros(T, size(xs)))
    res = convert(AT{Real}, zeros(T, size(xs, 1), size(xs, 1), size(xs, 2)))
    for i in axes(y, 1)
        z[i, :] .= one(T)
        res[i, :, :] .= only(back(z))
        z[i, :] .= zero(T)
    end
    y, res
end

function jacobian_batched(
    f,
    xs::AbstractMatrix{<:Real},
    T::Type{<:AbstractFloat},
    AT::Type{<:AbstractArray},
    CM::Type{<:SDVecJacMatrixMode},
)::Tuple
    y = f(xs)
    z = convert(AT, zeros(T, size(xs)))
    res = convert(AT{Real}, zeros(T, size(xs, 1), size(xs, 1), size(xs, 2)))
    for i in axes(y, 1)
        z[i, :] .= one(T)
        res[i, :, :] .= reshape(auto_vecjac(f, xs, z), size(xs))
        z[i, :] .= zero(T)
    end
    y, res
end

function jacobian_batched(
    f,
    xs::AbstractMatrix{<:Real},
    T::Type{<:AbstractFloat},
    AT::Type{<:AbstractArray},
    CM::Type{<:SDJacVecMatrixMode},
)::Tuple
    y = f(xs)
    z = convert(AT, zeros(T, size(xs)))
    res = convert(AT{Real}, zeros(T, size(xs, 1), size(xs, 1), size(xs, 2)))
    for i in axes(y, 1)
        z[i, :] .= one(T)
        res[i, :, :] .= reshape(auto_jacvec(f, xs, z), size(xs))
        z[i, :] .= zero(T)
    end
    y, res
end

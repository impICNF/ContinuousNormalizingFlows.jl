function jacobian_batched(f, xs::AbstractMatrix, T::Type{<: AbstractFloat}, AT::Type{<: AbstractArray})::Tuple
    y, back = Zygote.pullback(f, xs)
    z = convert(AT, zeros(T, size(xs)))
    res = convert(AT{Real}, zeros(T, size(xs, 1), size(xs, 1), size(xs, 2)))
    for i in 1:size(y, 1)
        z[i, :] .= one(T)
        res[i, :, :] .= only(back(z))
        z[i, :] .= zero(T)
    end
    y, res
end

function jacobian_batched(f, xs::AbstractMatrix, T::Type{<: AbstractFloat}, AT::Type{<: CuArray})::Tuple
    y, back = Zygote.pullback(f, xs)
    z = convert(AT, zeros(T, size(xs)))
    res = convert(AT, zeros(T, size(xs, 1), size(xs, 1), size(xs, 2)))
    for i in 1:size(y, 1)
        z[i, :] .= one(T)
        res[i, :, :] .= only(back(z))
        z[i, :] .= zero(T)
    end
    y, res
end

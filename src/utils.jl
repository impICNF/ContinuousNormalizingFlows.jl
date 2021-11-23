function jacobian_batched(f, x::AbstractMatrix{T}, move::MLJFlux.Mover)::Tuple where {T <: AbstractFloat}
    y, back = Zygote.pullback(f, x)
    z = zeros(eltype(x), size(x)) |> move
    res = zeros(size(x, 1), size(x, 1), size(x, 2)) |> move
    for i in 1:size(y, 1)
        z[i, :] .= one(eltype(x))
        res[i, :, :] .= only(back(z))
        z[i, :] .= zero(eltype(x))
    end
    y, res
end

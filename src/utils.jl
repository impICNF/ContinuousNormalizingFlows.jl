function jacobian_batched(f, xs::AbstractMatrix{T}, array_mover::Function)::Tuple where {T <: AbstractFloat}
    y, back = Zygote.pullback(f, xs)
    z = zeros(eltype(xs), size(xs)) |> array_mover
    res = zeros(size(xs, 1), size(xs, 1), size(xs, 2)) |> array_mover
    for i in 1:size(y, 1)
        z[i, :] .= one(eltype(xs))
        res[i, :, :] .= only(back(z))
        z[i, :] .= zero(eltype(xs))
    end
    y, res
end

function make_mover(acceleration::AbstractResource, data_type::DataType)
    (data_type <: AbstractFloat) || error("data_type must be a float type")

    if acceleration isa CUDALibs
        arr_t = CuArray
    else
        arr_t = Array
    end

    function mover(xs)
        convert(arr_t{data_type}, xs)
    end
    mover
end

function jacobian_batched(f, x::AbstractMatrix{T}, array_mover::Function)::Tuple where {T <: AbstractFloat}
    y, back = Zygote.pullback(f, x)
    z = zeros(eltype(x), size(x)) |> array_mover
    res = zeros(size(x, 1), size(x, 1), size(x, 2)) |> array_mover
    for i in 1:size(y, 1)
        z[i, :] .= one(eltype(x))
        res[i, :, :] .= only(back(z))
        z[i, :] .= zero(eltype(x))
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

    function mover(x)
        convert(arr_t{data_type}, x)
    end
    mover
end

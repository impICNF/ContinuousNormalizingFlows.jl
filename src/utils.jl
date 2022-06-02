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
    @assert acceleration isa Union{CPU1, CUDALibs}
    @assert data_type in DataType[Float64, Float32, Float16]

    if acceleration isa CPU1
        arr_t = Array
    elseif acceleration isa CUDALibs
        arr_t = CuArray
    end

    function mover(x)
        convert(arr_t{data_type}, x)
    end
    mover
end

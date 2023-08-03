function jacobian_batched(
    icnf::AbstractFlows{T, <:ZygoteMatrixMode},
    f,
    xs::AbstractMatrix{<:Real};
    resource::AbstractResource = icnf.resource,
) where {T <: AbstractFloat}
    y, back = Zygote.pullback(f, xs)
    z = zeros_T_AT(resource, icnf, size(xs))
    res = Zygote.Buffer(xs, size(xs, 1), size(xs, 1), size(xs, 2))
    for i in axes(xs, 1)
        @ignore_derivatives z[i, :] .= one(T)
        res[i, :, :] = only(back(z))
        @ignore_derivatives z[i, :] .= zero(T)
    end
    y, copy(res)
end

function jacobian_batched(
    icnf::AbstractFlows{T, <:SDVecJacMatrixMode},
    f,
    xs::AbstractMatrix{<:Real};
    resource::AbstractResource = icnf.resource,
) where {T <: AbstractFloat}
    y = f(xs)
    z = zeros_T_AT(resource, icnf, size(xs))
    res = Zygote.Buffer(xs, size(xs, 1), size(xs, 1), size(xs, 2))
    for i in axes(xs, 1)
        @ignore_derivatives z[i, :] .= one(T)
        res[i, :, :] = reshape(auto_vecjac(f, xs, z), size(xs))
        @ignore_derivatives z[i, :] .= zero(T)
    end
    y, copy(res)
end

function jacobian_batched(
    icnf::AbstractFlows{T, <:SDJacVecMatrixMode},
    f,
    xs::AbstractMatrix{<:Real};
    resource::AbstractResource = icnf.resource,
) where {T <: AbstractFloat}
    y = f(xs)
    z = zeros_T_AT(resource, icnf, size(xs))
    res = Zygote.Buffer(xs, size(xs, 1), size(xs, 1), size(xs, 2))
    for i in axes(xs, 1)
        @ignore_derivatives z[i, :] .= one(T)
        res[i, :, :] = reshape(auto_jacvec(f, xs, z), size(xs))
        @ignore_derivatives z[i, :] .= zero(T)
    end
    y, copy(res)
end

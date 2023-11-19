@views function jacobian_batched(
    icnf::AbstractFlows{T, <:SDVecJacMatrixMode},
    f,
    xs::AbstractMatrix{<:Real},
) where {T <: AbstractFloat}
    y = f(xs)
    z = similar(xs)
    @ignore_derivatives fill!(z, zero(T))
    res = Zygote.Buffer(xs, size(xs, 1), size(xs, 1), size(xs, 2))
    Jf = VecJac(f, xs; autodiff = icnf.autodiff_backend)
    for i in axes(xs, 1)
        @ignore_derivatives z[i, :] .= one(T)
        res[i, :, :] = reshape(Jf * z, size(xs))
        @ignore_derivatives z[i, :] .= zero(T)
    end
    y, copy(res)
end

@views function jacobian_batched(
    icnf::AbstractFlows{T, <:SDJacVecMatrixMode},
    f,
    xs::AbstractMatrix{<:Real},
) where {T <: AbstractFloat}
    y = f(xs)
    z = similar(xs)
    @ignore_derivatives fill!(z, zero(T))
    res = Zygote.Buffer(xs, size(xs, 1), size(xs, 1), size(xs, 2))
    Jf = JacVec(f, xs; autodiff = icnf.autodiff_backend)
    for i in axes(xs, 1)
        @ignore_derivatives z[i, :] .= one(T)
        res[i, :, :] = reshape(Jf * z, size(xs))
        @ignore_derivatives z[i, :] .= zero(T)
    end
    y, copy(res)
end

@views function jacobian_batched(
    ::AbstractFlows{T, <:ZygoteMatrixMode},
    f,
    xs::AbstractMatrix{<:Real},
) where {T <: AbstractFloat}
    y, VJ = Zygote.pullback(f, xs)
    z = similar(xs)
    @ignore_derivatives fill!(z, zero(T))
    res = Zygote.Buffer(xs, size(xs, 1), size(xs, 1), size(xs, 2))
    for i in axes(xs, 1)
        @ignore_derivatives z[i, :] .= one(T)
        res[i, :, :] = only(VJ(z))
        @ignore_derivatives z[i, :] .= zero(T)
    end
    y, copy(res)
end

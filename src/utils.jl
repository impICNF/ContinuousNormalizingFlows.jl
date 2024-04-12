@inline function jacobian_batched(
    icnf::AbstractICNF{T, <:DIVecJacMatrixMode},
    f::Function,
    xs::AbstractMatrix{<:Real},
) where {T}
    y, VJ = DifferentiationInterface.value_and_pullback_split(f, icnf.autodiff_backend, xs)
    z = similar(xs)
    @ignore_derivatives fill!(z, zero(T))
    res = Zygote.Buffer(xs, size(xs, 1), size(xs, 1), size(xs, 2))
    for i in axes(xs, 1)
        @ignore_derivatives z[i, :] .= one(T)
        res[i, :, :] = VJ(z)
        @ignore_derivatives z[i, :] .= zero(T)
    end
    y, copy(res)
end

@inline function jacobian_batched(
    icnf::AbstractICNF{T, <:DIMatrixMode},
    f::Function,
    xs::AbstractMatrix{<:Real},
) where {T}
    y, J = DifferentiationInterface.value_and_jacobian(f, icnf.autodiff_backend, xs)
    y, split_jac(J, size(xs, 1))
end

@inline function split_jac(x::AbstractMatrix{<:Real}, sz::Integer)
    (
        x[i:j, i:j] for (i, j) in zip(
            firstindex(x, 1):sz:lastindex(x, 1),
            (firstindex(x, 1) + sz - 1):sz:lastindex(x, 1),
        )
    )
end

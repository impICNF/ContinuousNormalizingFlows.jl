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

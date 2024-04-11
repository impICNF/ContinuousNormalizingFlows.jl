function jacobian_batched(
    icnf::AbstractICNF{T, <:DIMatrixMode},
    f::Function,
    xs::AbstractMatrix{<:Real},
) where {T}
    y, J = value_and_jacobian(f, icnf.autodiff_backend, xs)
    y, split_jac(J)
end

function split_jac(x::AbstractMatrix{<:Real})
    sz = convert(Int, sqrt(size(x, 1)))
    (x[i:(i + sz - 1), i:(i + sz - 1)] for i in firstindex(x, 1):sz:lastindex(x, 1))
end

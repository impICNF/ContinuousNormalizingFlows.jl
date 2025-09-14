struct Planar{
    T <: AbstractFloat,
    CM <: ComputeMode,
    INPLACE,
    COND,
    AUGMENTED,
    STEER,
    NORM_Z_AUG,
} <: AbstractICNF{T, CM, INPLACE, COND, AUGMENTED, STEER, NORM_Z_AUG} end
struct CondPlanar{
    T <: AbstractFloat,
    CM <: ComputeMode,
    INPLACE,
    COND,
    AUGMENTED,
    STEER,
    NORM_Z_AUG,
} <: AbstractICNF{T, CM, INPLACE, COND, AUGMENTED, STEER, NORM_Z_AUG} end

struct FFJORD{
    T <: AbstractFloat,
    CM <: ComputeMode,
    INPLACE,
    COND,
    AUGMENTED,
    STEER,
    NORM_Z_AUG,
} <: AbstractICNF{T, CM, INPLACE, COND, AUGMENTED, STEER, NORM_Z_AUG} end
struct CondFFJORD{
    T <: AbstractFloat,
    CM <: ComputeMode,
    INPLACE,
    COND,
    AUGMENTED,
    STEER,
    NORM_Z_AUG,
} <: AbstractICNF{T, CM, INPLACE, COND, AUGMENTED, STEER, NORM_Z_AUG} end

struct RNODE{
    T <: AbstractFloat,
    CM <: ComputeMode,
    INPLACE,
    COND,
    AUGMENTED,
    STEER,
    NORM_Z_AUG,
} <: AbstractICNF{T, CM, INPLACE, COND, AUGMENTED, STEER, NORM_Z_AUG} end
struct CondRNODE{
    T <: AbstractFloat,
    CM <: ComputeMode,
    INPLACE,
    COND,
    AUGMENTED,
    STEER,
    NORM_Z_AUG,
} <: AbstractICNF{T, CM, INPLACE, COND, AUGMENTED, STEER, NORM_Z_AUG} end

"""
Implementation of ICNF.

Refs:

[Chen, Ricky TQ, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. "Neural Ordinary Differential Equations." arXiv preprint arXiv:1806.07366 (2018).](https://arxiv.org/abs/1806.07366)

[Grathwohl, Will, Ricky TQ Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. "Ffjord: Free-form continuous dynamics for scalable reversible generative models." arXiv preprint arXiv:1810.01367 (2018).](https://arxiv.org/abs/1810.01367)

[Finlay, Chris, Jörn-Henrik Jacobsen, Levon Nurbekyan, and Adam M. Oberman. "How to train your neural ODE: the world of Jacobian and kinetic regularization." arXiv preprint arXiv:2002.02798 (2020).](https://arxiv.org/abs/2002.02798)
"""
struct ICNF{
    T <: AbstractFloat,
    CM <: ComputeMode,
    INPLACE,
    COND,
    AUGMENTED,
    STEER,
    NORM_Z,
    NORM_J,
    NORM_Z_AUG,
    NN <: LuxCore.AbstractLuxLayer,
    NVARS <: Int,
    DEVICE <: MLDataDevices.AbstractDevice,
    BASEDIST <: Distributions.Distribution,
    TSPAN <: NTuple{2, T},
    STEERDIST <: Distributions.Distribution,
    EPSDIST <: Distributions.Distribution,
    SOL_KWARGS <: NamedTuple,
    RNG <: Random.AbstractRNG,
} <: AbstractICNF{T, CM, INPLACE, COND, AUGMENTED, STEER, NORM_Z_AUG}
    nn::NN
    nvars::NVARS
    naugmented::NVARS

    compute_mode::CM
    device::DEVICE
    basedist::BASEDIST
    tspan::TSPAN
    steerdist::STEERDIST
    epsdist::EPSDIST
    sol_kwargs::SOL_KWARGS
    rng::RNG
    λ₁::T
    λ₂::T
    λ₃::T
end

function n_augment(::ICNF, ::Mode)
    return 2
end

function augmented_f(
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:DIVectorMode, false, COND, AUGMENTED, STEER, NORM_Z},
    mode::TestMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, REG}
    n_aug = n_augment(icnf, mode)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1)]
    ż, J = icnf_jacobian(icnf, mode, snn, z)
    l̇ = -LinearAlgebra.tr(J)
    Ė = if NORM_Z && REG
        LinearAlgebra.norm(ż)
    else
        zero(T)
    end
    ṅ = zero(T)
    return vcat(ż, l̇, Ė, ṅ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:DIVectorMode, true, COND, AUGMENTED, STEER, NORM_Z},
    mode::TestMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1)]
    ż, J = icnf_jacobian(icnf, mode, snn, z)
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -LinearAlgebra.tr(J)
    du[(end - n_aug + 1)] = if NORM_Z && REG
        LinearAlgebra.norm(ż)
    else
        zero(T)
    end
    du[(end - n_aug + 2)] = zero(T)
    return nothing
end

function augmented_f(
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:MatrixMode, false, COND, AUGMENTED, STEER, NORM_Z},
    mode::TestMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, J = icnf_jacobian(icnf, mode, snn, z)
    l̇ = -transpose(LinearAlgebra.tr.(eachslice(J; dims = 3)))
    Ė = transpose(if NORM_Z && REG
        LinearAlgebra.norm.(eachcol(ż))
    else
        zrs_Ė = similar(ż, size(ż, 2))
        ChainRulesCore.@ignore_derivatives fill!(zrs_Ė, zero(T))
        zrs_Ė
    end)
    ṅ = transpose(begin
        zrs_ṅ = similar(ż, size(ż, 2))
        ChainRulesCore.@ignore_derivatives fill!(zrs_ṅ, zero(T))
        zrs_ṅ
    end)
    return vcat(ż, l̇, Ė, ṅ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:MatrixMode, true, COND, AUGMENTED, STEER, NORM_Z},
    mode::TestMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, J = icnf_jacobian(icnf, mode, snn, z)
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -(LinearAlgebra.tr.(eachslice(J; dims = 3)))
    du[(end - n_aug + 1), :] .= if NORM_Z && REG
        LinearAlgebra.norm.(eachcol(ż))
    else
        zero(T)
    end
    du[(end - n_aug + 2), :] .= zero(T)
    return nothing
end

function augmented_f(
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:DIVecJacVectorMode, false, COND, AUGMENTED, STEER, NORM_Z, NORM_J},
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1)]
    ż, ϵJ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    l̇ = -LinearAlgebra.dot(ϵJ, ϵ)
    Ė = if NORM_Z && REG
        LinearAlgebra.norm(ż)
    else
        zero(T)
    end
    ṅ = if NORM_J && REG
        LinearAlgebra.norm(ϵJ)
    else
        zero(T)
    end
    return vcat(ż, l̇, Ė, ṅ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:DIVecJacVectorMode, true, COND, AUGMENTED, STEER, NORM_Z, NORM_J},
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1)]
    ż, ϵJ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -LinearAlgebra.dot(ϵJ, ϵ)
    du[(end - n_aug + 1)] = if NORM_Z && REG
        LinearAlgebra.norm(ż)
    else
        zero(T)
    end
    du[(end - n_aug + 2)] = if NORM_J && REG
        LinearAlgebra.norm(ϵJ)
    else
        zero(T)
    end
    return nothing
end

function augmented_f(
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:DIJacVecVectorMode, false, COND, AUGMENTED, STEER, NORM_Z, NORM_J},
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1)]
    ż, Jϵ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    l̇ = -LinearAlgebra.dot(ϵ, Jϵ)
    Ė = if NORM_Z && REG
        LinearAlgebra.norm(ż)
    else
        zero(T)
    end
    ṅ = if NORM_J && REG
        LinearAlgebra.norm(Jϵ)
    else
        zero(T)
    end
    return vcat(ż, l̇, Ė, ṅ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:DIJacVecVectorMode, true, COND, AUGMENTED, STEER, NORM_Z, NORM_J},
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1)]
    ż, Jϵ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -LinearAlgebra.dot(ϵ, Jϵ)
    du[(end - n_aug + 1)] = if NORM_Z && REG
        LinearAlgebra.norm(ż)
    else
        zero(T)
    end
    du[(end - n_aug + 2)] = if NORM_J && REG
        LinearAlgebra.norm(Jϵ)
    else
        zero(T)
    end
    return nothing
end

function augmented_f(
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:DIVecJacMatrixMode, false, COND, AUGMENTED, STEER, NORM_Z, NORM_J},
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, ϵJ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    l̇ = -sum(ϵJ .* ϵ; dims = 1)
    Ė = transpose(if NORM_Z && REG
        LinearAlgebra.norm.(eachcol(ż))
    else
        zrs_Ė = similar(ż, size(ż, 2))
        ChainRulesCore.@ignore_derivatives fill!(zrs_Ė, zero(T))
        zrs_Ė
    end)
    ṅ = transpose(if NORM_J && REG
        LinearAlgebra.norm.(eachcol(ϵJ))
    else
        zrs_ṅ = similar(ż, size(ż, 2))
        ChainRulesCore.@ignore_derivatives fill!(zrs_ṅ, zero(T))
        zrs_ṅ
    end)
    return vcat(ż, l̇, Ė, ṅ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:DIVecJacMatrixMode, true, COND, AUGMENTED, STEER, NORM_Z, NORM_J},
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, ϵJ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -vec(sum(ϵJ .* ϵ; dims = 1))
    du[(end - n_aug + 1), :] .= if NORM_Z && REG
        LinearAlgebra.norm.(eachcol(ż))
    else
        zero(T)
    end
    du[(end - n_aug + 2), :] .= if NORM_J && REG
        LinearAlgebra.norm.(eachcol(ϵJ))
    else
        zero(T)
    end
    return nothing
end

function augmented_f(
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:DIJacVecMatrixMode, false, COND, AUGMENTED, STEER, NORM_Z, NORM_J},
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, Jϵ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    l̇ = -sum(ϵ .* Jϵ; dims = 1)
    Ė = transpose(if NORM_Z && REG
        LinearAlgebra.norm.(eachcol(ż))
    else
        zrs_Ė = similar(ż, size(ż, 2))
        ChainRulesCore.@ignore_derivatives fill!(zrs_Ė, zero(T))
        zrs_Ė
    end)
    ṅ = transpose(if NORM_J && REG
        LinearAlgebra.norm.(eachcol(Jϵ))
    else
        zrs_ṅ = similar(ż, size(ż, 2))
        ChainRulesCore.@ignore_derivatives fill!(zrs_ṅ, zero(T))
        zrs_ṅ
    end)
    return vcat(ż, l̇, Ė, ṅ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:DIJacVecMatrixMode, true, COND, AUGMENTED, STEER, NORM_Z, NORM_J},
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, Jϵ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -vec(sum(ϵ .* Jϵ; dims = 1))
    du[(end - n_aug + 1), :] .= if NORM_Z && REG
        LinearAlgebra.norm.(eachcol(ż))
    else
        zero(T)
    end
    du[(end - n_aug + 2), :] .= if NORM_J && REG
        LinearAlgebra.norm.(eachcol(Jϵ))
    else
        zero(T)
    end
    return nothing
end

function augmented_f(
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:LuxVecJacMatrixMode, false, COND, AUGMENTED, STEER, NORM_Z, NORM_J},
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, ϵJ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    l̇ = -sum(ϵJ .* ϵ; dims = 1)
    Ė = transpose(if NORM_Z && REG
        LinearAlgebra.norm.(eachcol(ż))
    else
        zrs_Ė = similar(ż, size(ż, 2))
        ChainRulesCore.@ignore_derivatives fill!(zrs_Ė, zero(T))
        zrs_Ė
    end)
    ṅ = transpose(if NORM_J && REG
        LinearAlgebra.norm.(eachcol(ϵJ))
    else
        zrs_ṅ = similar(ż, size(ż, 2))
        ChainRulesCore.@ignore_derivatives fill!(zrs_ṅ, zero(T))
        zrs_ṅ
    end)
    return vcat(ż, l̇, Ė, ṅ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:LuxVecJacMatrixMode, true, COND, AUGMENTED, STEER, NORM_Z, NORM_J},
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, ϵJ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -vec(sum(ϵJ .* ϵ; dims = 1))
    du[(end - n_aug + 1), :] .= if NORM_Z && REG
        LinearAlgebra.norm.(eachcol(ż))
    else
        zero(T)
    end
    du[(end - n_aug + 2), :] .= if NORM_J && REG
        LinearAlgebra.norm.(eachcol(ϵJ))
    else
        zero(T)
    end
    return nothing
end

function augmented_f(
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:LuxJacVecMatrixMode, false, COND, AUGMENTED, STEER, NORM_Z, NORM_J},
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, Jϵ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    l̇ = -sum(ϵ .* Jϵ; dims = 1)
    Ė = transpose(if NORM_Z && REG
        LinearAlgebra.norm.(eachcol(ż))
    else
        zrs_Ė = similar(ż, size(ż, 2))
        ChainRulesCore.@ignore_derivatives fill!(zrs_Ė, zero(T))
        zrs_Ė
    end)
    ṅ = transpose(if NORM_J && REG
        LinearAlgebra.norm.(eachcol(Jϵ))
    else
        zrs_ṅ = similar(ż, size(ż, 2))
        ChainRulesCore.@ignore_derivatives fill!(zrs_ṅ, zero(T))
        zrs_ṅ
    end)
    return vcat(ż, l̇, Ė, ṅ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:LuxJacVecMatrixMode, true, COND, AUGMENTED, STEER, NORM_Z, NORM_J},
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, Jϵ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -vec(sum(ϵ .* Jϵ; dims = 1))
    du[(end - n_aug + 1), :] .= if NORM_Z && REG
        LinearAlgebra.norm.(eachcol(ż))
    else
        zero(T)
    end
    du[(end - n_aug + 2), :] .= if NORM_J && REG
        LinearAlgebra.norm.(eachcol(Jϵ))
    else
        zero(T)
    end
    return nothing
end

function loss(
    icnf::ICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::NamedTuple,
)
    logp̂x, (Ė, ṅ, Ȧ) = inference(icnf, mode, xs, ps, st)
    return -logp̂x + icnf.λ₁ * Ė + icnf.λ₂ * ṅ + icnf.λ₃ * Ȧ
end

function loss(
    icnf::ICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::NamedTuple,
)
    logp̂x, (Ė, ṅ, Ȧ) = inference(icnf, mode, xs, ys, ps, st)
    return -logp̂x + icnf.λ₁ * Ė + icnf.λ₂ * ṅ + icnf.λ₃ * Ȧ
end

function loss(
    icnf::ICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::NamedTuple,
)
    logp̂x, (Ė, ṅ, Ȧ) = inference(icnf, mode, xs, ps, st)
    return Statistics.mean(-logp̂x + icnf.λ₁ * Ė + icnf.λ₂ * ṅ + icnf.λ₃ * Ȧ)
end

function loss(
    icnf::ICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::NamedTuple,
)
    logp̂x, (Ė, ṅ, Ȧ) = inference(icnf, mode, xs, ys, ps, st)
    return Statistics.mean(-logp̂x + icnf.λ₁ * Ė + icnf.λ₂ * ṅ + icnf.λ₃ * Ȧ)
end

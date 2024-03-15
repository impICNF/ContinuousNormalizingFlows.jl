export ICNF, RNODE, CondRNODE, FFJORD, CondFFJORD, Planar, CondPlanar

struct Planar end
struct CondPlanar end

struct FFJORD end
struct CondFFJORD end

struct RNODE end
struct CondRNODE end

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
    NN <: LuxCore.AbstractExplicitLayer,
    NVARS <: Int,
    RESOURCE <: AbstractResource,
    BASEDIST <: Distribution,
    TSPAN <: NTuple{2, T},
    STEERDIST <: Distribution,
    EPSDIST <: Distribution,
    DIFFERENTIATION_BACKEND <: AbstractDifferentiation.AbstractBackend,
    AUTODIFF_BACKEND <: ADTypes.AbstractADType,
    SOL_KWARGS <: NamedTuple,
    RNG <: AbstractRNG,
} <: AbstractICNF{T, CM, INPLACE, COND, AUGMENTED, STEER}
    nn::NN
    nvars::NVARS
    naugmented::NVARS

    resource::RESOURCE
    basedist::BASEDIST
    tspan::TSPAN
    steerdist::STEERDIST
    epsdist::EPSDIST
    differentiation_backend::DIFFERENTIATION_BACKEND
    autodiff_backend::AUTODIFF_BACKEND
    sol_kwargs::SOL_KWARGS
    rng::RNG
    λ₁::T
    λ₂::T
    λ₃::T
end

@inline function n_augment(::ICNF, ::TrainMode)
    3
end

function augmented_f(
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:ADVectorMode, false},
    mode::TestMode,
    nn,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, J = AbstractDifferentiation.value_and_jacobian(
        icnf.differentiation_backend,
        let p = p
            x -> nn(x, p)
        end,
        z,
    )
    l̇ = -tr(only(J))
    vcat(ż, l̇)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:ADVectorMode, true},
    mode::TestMode,
    nn,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, J = AbstractDifferentiation.value_and_jacobian(
        icnf.differentiation_backend,
        let p = p
            x -> nn(x, p)
        end,
        z,
    )
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -tr(only(J))
    nothing
end

function augmented_f(
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:ZygoteVectorMode, false},
    mode::TestMode,
    nn,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, J = Zygote.withjacobian(let p = p
        x -> nn(x, p)
    end, z)
    l̇ = -tr(only(J))
    vcat(ż, l̇)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:ZygoteVectorMode, true},
    mode::TestMode,
    nn,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, J = Zygote.withjacobian(let p = p
        x -> nn(x, p)
    end, z)
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -tr(only(J))
    nothing
end

function augmented_f(
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:MatrixMode, false},
    mode::TestMode,
    nn,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż, J = jacobian_batched(icnf, let p = p
        x -> nn(x, p)
    end, z)
    l̇ = -transpose(tr.(eachslice(J; dims = 3)))
    vcat(ż, l̇)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{T, <:MatrixMode, true},
    mode::TestMode,
    nn,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż, J = jacobian_batched(icnf, let p = p
        x -> nn(x, p)
    end, z)
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -(tr.(eachslice(J; dims = 3)))
    nothing
end

function augmented_f(
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{
        T,
        <:ADVecJacVectorMode,
        false,
        COND,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
        NORM_Z_AUG,
    },
    mode::TrainMode,
    nn,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, NORM_Z_AUG}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    if AUGMENTED
        n_aug_input = n_augment_input(icnf)
        z_aug = z[(end - n_aug_input + 1):end]
    end
    ż, VJ = AbstractDifferentiation.value_and_pullback_function(
        icnf.differentiation_backend,
        let p = p
            x -> nn(x, p)
        end,
        z,
    )
    ϵJ = only(VJ(ϵ))
    l̇ = -(ϵJ ⋅ ϵ)
    Ė = if NORM_Z
        norm(ż)
    else
        zero(T)
    end
    ṅ = if NORM_J
        norm(ϵJ)
    else
        zero(T)
    end
    Ȧ = if (NORM_Z_AUG && AUGMENTED)
        norm(z_aug)
    else
        zero(T)
    end
    vcat(ż, l̇, Ė, ṅ, Ȧ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{
        T,
        <:ADVecJacVectorMode,
        true,
        COND,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
        NORM_Z_AUG,
    },
    mode::TrainMode,
    nn,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, NORM_Z_AUG}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    if AUGMENTED
        n_aug_input = n_augment_input(icnf)
        z_aug = z[(end - n_aug_input + 1):end]
    end
    ż, VJ = AbstractDifferentiation.value_and_pullback_function(
        icnf.differentiation_backend,
        let p = p
            x -> nn(x, p)
        end,
        z,
    )
    ϵJ = only(VJ(ϵ))
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -(ϵJ ⋅ ϵ)
    du[(end - n_aug + 1)] = if NORM_Z
        norm(ż)
    else
        zero(T)
    end
    du[(end - n_aug + 2)] = if NORM_J
        norm(ϵJ)
    else
        zero(T)
    end
    du[(end - n_aug + 3)] = if (NORM_Z_AUG && AUGMENTED)
        norm(z_aug)
    else
        zero(T)
    end
    nothing
end

function augmented_f(
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{
        T,
        <:ADJacVecVectorMode,
        false,
        COND,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
        NORM_Z_AUG,
    },
    mode::TrainMode,
    nn,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, NORM_Z_AUG}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    if AUGMENTED
        n_aug_input = n_augment_input(icnf)
        z_aug = z[(end - n_aug_input + 1):end]
    end
    ż_JV = AbstractDifferentiation.value_and_pushforward_function(
        icnf.differentiation_backend,
        let p = p
            x -> nn(x, p)
        end,
        z,
    )
    ż, Jϵ = ż_JV(ϵ)
    Jϵ = only(Jϵ)
    l̇ = -(ϵ ⋅ Jϵ)
    Ė = if NORM_Z
        norm(ż)
    else
        zero(T)
    end
    ṅ = if NORM_J
        norm(Jϵ)
    else
        zero(T)
    end
    Ȧ = if NORM_Z_AUG
        norm(z_aug)
    else
        zero(T)
    end
    vcat(ż, l̇, Ė, ṅ, Ȧ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{
        T,
        <:ADJacVecVectorMode,
        true,
        COND,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
        NORM_Z_AUG,
    },
    mode::TrainMode,
    nn,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, NORM_Z_AUG}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    if AUGMENTED
        n_aug_input = n_augment_input(icnf)
        z_aug = z[(end - n_aug_input + 1):end]
    end
    ż_JV = AbstractDifferentiation.value_and_pushforward_function(
        icnf.differentiation_backend,
        let p = p
            x -> nn(x, p)
        end,
        z,
    )
    ż, Jϵ = ż_JV(ϵ)
    Jϵ = only(Jϵ)
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -(ϵ ⋅ Jϵ)
    du[(end - n_aug + 1)] = if NORM_Z
        norm(ż)
    else
        zero(T)
    end
    du[(end - n_aug + 2)] = if NORM_J
        norm(Jϵ)
    else
        zero(T)
    end
    du[(end - n_aug + 3)] = if (NORM_Z_AUG && AUGMENTED)
        norm(z_aug)
    else
        zero(T)
    end
    nothing
end

function augmented_f(
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{
        T,
        <:ZygoteVectorMode,
        false,
        COND,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
        NORM_Z_AUG,
    },
    mode::TrainMode,
    nn,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, NORM_Z_AUG}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    if AUGMENTED
        n_aug_input = n_augment_input(icnf)
        z_aug = z[(end - n_aug_input + 1):end]
    end
    ż, VJ = Zygote.pullback(let p = p
        x -> nn(x, p)
    end, z)
    ϵJ = only(VJ(ϵ))
    l̇ = -(ϵJ ⋅ ϵ)
    Ė = if NORM_Z
        norm(ż)
    else
        zero(T)
    end
    ṅ = if NORM_J
        norm(ϵJ)
    else
        zero(T)
    end
    Ȧ = if (NORM_Z_AUG && AUGMENTED)
        norm(z_aug)
    else
        zero(T)
    end
    vcat(ż, l̇, Ė, ṅ, Ȧ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{
        T,
        <:ZygoteVectorMode,
        true,
        COND,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
        NORM_Z_AUG,
    },
    mode::TrainMode,
    nn,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, NORM_Z_AUG}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    if AUGMENTED
        n_aug_input = n_augment_input(icnf)
        z_aug = z[(end - n_aug_input + 1):end]
    end
    ż, VJ = Zygote.pullback(let p = p
        x -> nn(x, p)
    end, z)
    ϵJ = only(VJ(ϵ))
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -(ϵJ ⋅ ϵ)
    du[(end - n_aug + 1)] = if NORM_Z
        norm(ż)
    else
        zero(T)
    end
    du[(end - n_aug + 2)] = if NORM_J
        norm(ϵJ)
    else
        zero(T)
    end
    du[(end - n_aug + 3)] = if (NORM_Z_AUG && AUGMENTED)
        norm(z_aug)
    else
        zero(T)
    end
    nothing
end

function augmented_f(
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{
        T,
        <:SDVecJacMatrixMode,
        false,
        COND,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
        NORM_Z_AUG,
    },
    mode::TrainMode,
    nn,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, NORM_Z_AUG}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    if AUGMENTED
        n_aug_input = n_augment_input(icnf)
        z_aug = z[(end - n_aug_input + 1):end, :]
    end
    ż = nn(z, p)
    Jf = VecJac(let p = p
        x -> nn(x, p)
    end, z; autodiff = icnf.autodiff_backend)
    ϵJ = reshape(Jf * ϵ, size(z))
    l̇ = -sum(ϵJ .* ϵ; dims = 1)
    Ė = transpose(if NORM_Z
        norm.(eachcol(ż))
    else
        zeros(T, size(u, 2))
    end)
    ṅ = transpose(if NORM_J
        norm.(eachcol(ϵJ))
    else
        zeros(T, size(u, 2))
    end)
    Ȧ = transpose(if (NORM_Z_AUG && AUGMENTED)
        norm.(eachcol(z_aug))
    else
        zeros(T, size(u, 2))
    end)
    vcat(ż, l̇, Ė, ṅ, Ȧ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{
        T,
        <:SDVecJacMatrixMode,
        true,
        COND,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
        NORM_Z_AUG,
    },
    mode::TrainMode,
    nn,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, NORM_Z_AUG}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    if AUGMENTED
        n_aug_input = n_augment_input(icnf)
        z_aug = z[(end - n_aug_input + 1):end, :]
    end
    ż = nn(z, p)
    Jf = VecJac(let p = p
        x -> nn(x, p)
    end, z; autodiff = icnf.autodiff_backend)
    ϵJ = reshape(Jf * ϵ, size(z))
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -vec(sum(ϵJ .* ϵ; dims = 1))
    du[(end - n_aug + 1), :] .= if NORM_Z
        norm.(eachcol(ż))
    else
        zeros(T, size(u, 2))
    end
    du[(end - n_aug + 2), :] .= if NORM_J
        norm.(eachcol(ϵJ))
    else
        zeros(T, size(u, 2))
    end
    du[(end - n_aug + 3), :] .= if (NORM_Z_AUG && AUGMENTED)
        norm.(eachcol(z_aug))
    else
        zeros(T, size(u, 2))
    end
    nothing
end

function augmented_f(
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{
        T,
        <:SDJacVecMatrixMode,
        false,
        COND,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
        NORM_Z_AUG,
    },
    mode::TrainMode,
    nn,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, NORM_Z_AUG}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    if AUGMENTED
        n_aug_input = n_augment_input(icnf)
        z_aug = z[(end - n_aug_input + 1):end, :]
    end
    ż = nn(z, p)
    Jf = JacVec(let p = p
        x -> nn(x, p)
    end, z; autodiff = icnf.autodiff_backend)
    Jϵ = reshape(Jf * ϵ, size(z))
    l̇ = -sum(ϵ .* Jϵ; dims = 1)
    Ė = transpose(if NORM_Z
        norm.(eachcol(ż))
    else
        zeros(T, size(u, 2))
    end)
    ṅ = transpose(if NORM_J
        norm.(eachcol(Jϵ))
    else
        zeros(T, size(u, 2))
    end)
    Ȧ = transpose(if (NORM_Z_AUG && AUGMENTED)
        norm.(eachcol(z_aug))
    else
        zeros(T, size(u, 2))
    end)
    vcat(ż, l̇, Ė, ṅ, Ȧ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{
        T,
        <:SDJacVecMatrixMode,
        true,
        COND,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
        NORM_Z_AUG,
    },
    mode::TrainMode,
    nn,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, NORM_Z_AUG}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    if AUGMENTED
        n_aug_input = n_augment_input(icnf)
        z_aug = z[(end - n_aug_input + 1):end, :]
    end
    ż = nn(z, p)
    Jf = JacVec(let p = p
        x -> nn(x, p)
    end, z; autodiff = icnf.autodiff_backend)
    Jϵ = reshape(Jf * ϵ, size(z))
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -vec(sum(ϵ .* Jϵ; dims = 1))
    du[(end - n_aug + 1), :] .= if NORM_Z
        norm.(eachcol(ż))
    else
        zeros(T, size(u, 2))
    end
    du[(end - n_aug + 2), :] .= if NORM_J
        norm.(eachcol(Jϵ))
    else
        zeros(T, size(u, 2))
    end
    du[(end - n_aug + 3), :] .= if (NORM_Z_AUG && AUGMENTED)
        norm.(eachcol(z_aug))
    else
        zeros(T, size(u, 2))
    end
    nothing
end

function augmented_f(
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{
        T,
        <:ZygoteMatrixMode,
        false,
        COND,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
        NORM_Z_AUG,
    },
    mode::TrainMode,
    nn,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, NORM_Z_AUG}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    if AUGMENTED
        n_aug_input = n_augment_input(icnf)
        z_aug = z[(end - n_aug_input + 1):end, :]
    end
    ż, VJ = Zygote.pullback(let p = p
        x -> nn(x, p)
    end, z)
    ϵJ = only(VJ(ϵ))
    l̇ = -sum(ϵJ .* ϵ; dims = 1)
    Ė = transpose(if NORM_Z
        norm.(eachcol(ż))
    else
        zeros(T, size(u, 2))
    end)
    ṅ = transpose(if NORM_J
        norm.(eachcol(ϵJ))
    else
        zeros(T, size(u, 2))
    end)
    Ȧ = transpose(if (NORM_Z_AUG && AUGMENTED)
        norm.(eachcol(z_aug))
    else
        zeros(T, size(u, 2))
    end)
    vcat(ż, l̇, Ė, ṅ, Ȧ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    ::Any,
    icnf::ICNF{
        T,
        <:ZygoteMatrixMode,
        true,
        COND,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
        NORM_Z_AUG,
    },
    mode::TrainMode,
    nn,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUGMENTED, STEER, NORM_Z, NORM_J, NORM_Z_AUG}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    if AUGMENTED
        n_aug_input = n_augment_input(icnf)
        z_aug = z[(end - n_aug_input + 1):end, :]
    end
    ż, VJ = Zygote.pullback(let p = p
        x -> nn(x, p)
    end, z)
    ϵJ = only(VJ(ϵ))
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -vec(sum(ϵJ .* ϵ; dims = 1))
    du[(end - n_aug + 1), :] .= if NORM_Z
        norm.(eachcol(ż))
    else
        zeros(T, size(u, 2))
    end
    du[(end - n_aug + 2), :] .= if NORM_J
        norm.(eachcol(ϵJ))
    else
        zeros(T, size(u, 2))
    end
    du[(end - n_aug + 3), :] .= if (NORM_Z_AUG && AUGMENTED)
        norm.(eachcol(z_aug))
    else
        zeros(T, size(u, 2))
    end
    nothing
end

@inline function loss(
    icnf::ICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ps::Any,
)
    logp̂x, (Ė, ṅ, Ȧ) = inference(icnf, mode, xs, ps)
    -logp̂x + icnf.λ₁ * Ė + icnf.λ₂ * ṅ + icnf.λ₃ * Ȧ
end

@inline function loss(
    icnf::ICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
)
    logp̂x, (Ė, ṅ, Ȧ) = inference(icnf, mode, xs, ps)
    mean(-logp̂x + icnf.λ₁ * Ė + icnf.λ₂ * ṅ + icnf.λ₃ * Ȧ)
end

"""
Implementation of ICNF.

Refs:

[Chen, Ricky TQ, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. "Neural Ordinary Differential Equations." arXiv preprint arXiv:1806.07366 (2018).](https://arxiv.org/abs/1806.07366)

[Grathwohl, Will, Ricky TQ Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. "Ffjord: Free-form continuous dynamics for scalable reversible generative models." arXiv preprint arXiv:1810.01367 (2018).](https://arxiv.org/abs/1810.01367)

[Finlay, Chris, Jörn-Henrik Jacobsen, Levon Nurbekyan, and Adam M. Oberman. "How to train your neural ODE: the world of Jacobian and kinetic regularization." arXiv preprint arXiv:2002.02798 (2020).](https://arxiv.org/abs/2002.02798)

[Dupont, Emilien, Arnaud Doucet, and Yee Whye Teh. "Augmented Neural ODEs." arXiv preprint arXiv:1904.01681 (2019).](https://arxiv.org/abs/1904.01681)

[Ghosh, Arnab, Harkirat Singh Behl, Emilien Dupont, Philip HS Torr, and Vinay Namboodiri. "STEER: Simple Temporal Regularization For Neural ODEs." arXiv preprint arXiv:2006.10711 (2020).](https://arxiv.org/abs/2006.10711)
"""
struct ICNF{
    T <: AbstractFloat,
    CM <: ComputeMode,
    INPLACE,
    COND,
    AUTONOMOUS,
    AUGMENTED,
    STEER,
    NORM_Z,
    NORM_J,
    NORM_Z_AUG,
    DEVICE <: MLDataDevices.AbstractDevice,
    RNG <: Random.AbstractRNG,
    TSPAN <: NTuple{2, T},
    NVARS <: Int,
    NN <: LuxCore.AbstractLuxLayer,
    BASEDIST <: Distributions.Distribution,
    EPSDIST <: Distributions.Distribution,
    STEERDIST <: Distributions.Distribution,
    SOL_KWARGS <: NamedTuple,
} <: AbstractICNF{T, CM, INPLACE, COND, AUGMENTED, STEER, NORM_Z_AUG}
    compute_mode::CM
    device::DEVICE
    rng::RNG
    tspan::TSPAN
    nvars::NVARS
    naugmented::NVARS
    nn::NN
    λ₁::T
    λ₂::T
    λ₃::T
    basedist::BASEDIST
    epsdist::EPSDIST
    steerdist::STEERDIST
    sol_kwargs::SOL_KWARGS
end

function ICNF(;
    data_type::Type{<:AbstractFloat} = Float32,
    compute_mode::ComputeMode = LuxVecJacMatrixMode(ADTypes.AutoZygote()),
    inplace::Bool = false,
    cond::Bool = false,
    autonomous::Bool = false,
    device::MLDataDevices.AbstractDevice = MLDataDevices.cpu_device(),
    rng::Random.AbstractRNG = MLDataDevices.default_device_rng(device),
    tspan::NTuple{2} = (zero(data_type), one(data_type)),
    nvars::Int = 1,
    naugmented::Int = nvars + 1,
    nn::LuxCore.AbstractLuxLayer = Lux.Chain(
        Lux.Dense(nvars + naugmented + !autonomous => nvars + naugmented, tanh),
    ),
    steer_rate::AbstractFloat = convert(data_type, 1.0e-1),
    λ₁::AbstractFloat = convert(data_type, 1.0e-2),
    λ₂::AbstractFloat = convert(data_type, 1.0e-2),
    λ₃::AbstractFloat = convert(data_type, 1.0e-2),
    basedist::Distributions.Distribution = Distributions.MvNormal(
        FillArrays.Zeros{data_type}(nvars + naugmented),
        FillArrays.Eye{data_type}(nvars + naugmented),
    ),
    epsdist::Distributions.Distribution = Distributions.MvNormal(
        FillArrays.Zeros{data_type}(nvars + naugmented),
        FillArrays.Eye{data_type}(nvars + naugmented),
    ),
    sol_kwargs::NamedTuple = (;
        save_everystep = false,
        maxiters = typemax(Int),
        reltol = convert(data_type, 1.0e-4),
        abstol = convert(data_type, 1.0e-8),
        alg = OrdinaryDiffEqAdamsBashforthMoulton.VCABM(; thread = Static.True()),
        sensealg = SciMLSensitivity.InterpolatingAdjoint(;
            checkpointing = true,
            autodiff = true,
        ),
    ),
)
    steerdist = Distributions.Uniform{data_type}(-steer_rate, steer_rate)
    return ICNF{
        data_type,
        typeof(compute_mode),
        inplace,
        cond,
        autonomous,
        !iszero(naugmented),
        !iszero(steer_rate),
        !iszero(λ₁),
        !iszero(λ₂),
        !iszero(λ₃),
        typeof(device),
        typeof(rng),
        typeof(tspan),
        typeof(nvars),
        typeof(nn),
        typeof(basedist),
        typeof(epsdist),
        typeof(steerdist),
        typeof(sol_kwargs),
    }(
        compute_mode,
        device,
        rng,
        tspan,
        nvars,
        naugmented,
        nn,
        λ₁,
        λ₂,
        λ₃,
        basedist,
        epsdist,
        steerdist,
        sol_kwargs,
    )
end

function n_augment(::ICNF, ::Mode)
    return 2
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::ICNF{T, <:DIVectorMode, false, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z},
    mode::TestMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z, REG}
    n_aug = n_augment(icnf, mode)
    nn = ifelse(AUTONOMOUS, nn, CondLayer(nn, t))
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
    t::Any,
    icnf::ICNF{T, <:DIVectorMode, true, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z},
    mode::TestMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z, REG}
    n_aug = n_augment(icnf, mode)
    nn = ifelse(AUTONOMOUS, nn, CondLayer(nn, t))
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
    t::Any,
    icnf::ICNF{T, <:MatrixMode, false, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z},
    mode::TestMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z, REG}
    n_aug = n_augment(icnf, mode)
    nn = ifelse(AUTONOMOUS, nn, CondLayer(nn, t))
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
    t::Any,
    icnf::ICNF{T, <:MatrixMode, true, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z},
    mode::TestMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z, REG}
    n_aug = n_augment(icnf, mode)
    nn = ifelse(AUTONOMOUS, nn, CondLayer(nn, t))
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
    t::Any,
    icnf::ICNF{
        T,
        <:DIVecJacVectorMode,
        false,
        COND,
        AUTONOMOUS,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
    },
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    nn = ifelse(AUTONOMOUS, nn, CondLayer(nn, t))
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
    t::Any,
    icnf::ICNF{
        T,
        <:DIVecJacVectorMode,
        true,
        COND,
        AUTONOMOUS,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
    },
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    nn = ifelse(AUTONOMOUS, nn, CondLayer(nn, t))
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
    t::Any,
    icnf::ICNF{
        T,
        <:DIJacVecVectorMode,
        false,
        COND,
        AUTONOMOUS,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
    },
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    nn = ifelse(AUTONOMOUS, nn, CondLayer(nn, t))
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
    t::Any,
    icnf::ICNF{
        T,
        <:DIJacVecVectorMode,
        true,
        COND,
        AUTONOMOUS,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
    },
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    nn = ifelse(AUTONOMOUS, nn, CondLayer(nn, t))
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
    t::Any,
    icnf::ICNF{
        T,
        <:DIVecJacMatrixMode,
        false,
        COND,
        AUTONOMOUS,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
    },
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    nn = ifelse(AUTONOMOUS, nn, CondLayer(nn, t))
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
    t::Any,
    icnf::ICNF{
        T,
        <:DIVecJacMatrixMode,
        true,
        COND,
        AUTONOMOUS,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
    },
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    nn = ifelse(AUTONOMOUS, nn, CondLayer(nn, t))
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
    t::Any,
    icnf::ICNF{
        T,
        <:DIJacVecMatrixMode,
        false,
        COND,
        AUTONOMOUS,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
    },
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    nn = ifelse(AUTONOMOUS, nn, CondLayer(nn, t))
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
    t::Any,
    icnf::ICNF{
        T,
        <:DIJacVecMatrixMode,
        true,
        COND,
        AUTONOMOUS,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
    },
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    nn = ifelse(AUTONOMOUS, nn, CondLayer(nn, t))
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
    t::Any,
    icnf::ICNF{
        T,
        <:LuxVecJacMatrixMode,
        false,
        COND,
        AUTONOMOUS,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
    },
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    nn = ifelse(AUTONOMOUS, nn, CondLayer(nn, t))
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
    t::Any,
    icnf::ICNF{
        T,
        <:LuxVecJacMatrixMode,
        true,
        COND,
        AUTONOMOUS,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
    },
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    nn = ifelse(AUTONOMOUS, nn, CondLayer(nn, t))
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
    t::Any,
    icnf::ICNF{
        T,
        <:LuxJacVecMatrixMode,
        false,
        COND,
        AUTONOMOUS,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
    },
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    nn = ifelse(AUTONOMOUS, nn, CondLayer(nn, t))
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
    t::Any,
    icnf::ICNF{
        T,
        <:LuxJacVecMatrixMode,
        true,
        COND,
        AUTONOMOUS,
        AUGMENTED,
        STEER,
        NORM_Z,
        NORM_J,
    },
    mode::TrainMode{REG},
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat, COND, AUTONOMOUS, AUGMENTED, STEER, NORM_Z, NORM_J, REG}
    n_aug = n_augment(icnf, mode)
    nn = ifelse(AUTONOMOUS, nn, CondLayer(nn, t))
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

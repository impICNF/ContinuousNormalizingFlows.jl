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
    CONDITIONED,
    AUTONOMOUS,
    AUGMENTED,
    STEER,
    NORM_Z,
    NORM_J,
    NORM_Z_AUG,
    DEVICE <: MLDataDevices.AbstractDevice,
    RNG <: Random.AbstractRNG,
    TSPAN <: NTuple{2, T},
    NVARIABLES <: Int,
    NN <: LuxCore.AbstractLuxLayer,
    BASEDIST <: Distributions.Distribution,
    EPSDIST <: Distributions.Distribution,
    STEERDIST <: Distributions.Distribution,
    SOL_KWARGS <: NamedTuple,
} <: AbstractICNF{T, CM, INPLACE, CONDITIONED, AUGMENTED, STEER, NORM_Z_AUG}
    compute_mode::CM
    device::DEVICE
    rng::RNG
    tspan::TSPAN
    nvariables::NVARIABLES
    naugments::NVARIABLES
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
    autonomous::Bool = false,
    device::MLDataDevices.AbstractDevice = MLDataDevices.cpu_device(),
    rng::Random.AbstractRNG = MLDataDevices.default_device_rng(device),
    tspan::NTuple{2} = (zero(data_type), one(data_type)),
    nvariables::Int = 1,
    naugments::Int = nvariables + 1,
    nconditions::Int = 0,
    n_in::Int = nvariables + naugments + !autonomous + nconditions,
    n_out::Int = nvariables + naugments,
    n_hidden::Int = n_in * 4,
    nn::LuxCore.AbstractLuxLayer = Lux.Chain(
        Lux.Dense(n_in => n_hidden, NNlib.softplus),
        Lux.Dense(n_hidden => n_hidden, NNlib.softplus),
        Lux.Dense(n_hidden => n_out),
    ),
    steer_rate::AbstractFloat = convert(data_type, 1.0e-1),
    λ₁::AbstractFloat = convert(data_type, 1.0e-2),
    λ₂::AbstractFloat = convert(data_type, 1.0e-2),
    λ₃::AbstractFloat = convert(data_type, 1.0e-2),
    basedist::Distributions.Distribution = Distributions.MvNormal(
        FillArrays.Zeros{data_type}(nvariables + naugments),
        FillArrays.Eye{data_type}(nvariables + naugments),
    ),
    epsdist::Distributions.Distribution = Distributions.MvNormal(
        FillArrays.Zeros{data_type}(nvariables + naugments),
        FillArrays.Eye{data_type}(nvariables + naugments),
    ),
    sol_kwargs::NamedTuple = (;
        save_everystep = false,
        maxiters = typemax(Int),
        reltol = sqrt(eps(data_type)),
        abstol = sqrt(eps(data_type)),
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
        !iszero(nconditions),
        autonomous,
        !iszero(naugments),
        !iszero(steer_rate),
        !iszero(λ₁),
        !iszero(λ₂),
        !iszero(λ₃),
        typeof(device),
        typeof(rng),
        typeof(tspan),
        typeof(nvariables),
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
        nvariables,
        naugments,
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

function n_augments(::ICNF, ::Mode)
    return 2
end

function add_time_nn(
    ::ICNF{<:AbstractFloat, <:ComputeMode, INPLACE, CONDITIONED, false},
    nn::LuxCore.AbstractLuxLayer,
    t::Number,
) where {INPLACE, CONDITIONED}
    return CondLayer(nn, t)
end

function add_time_nn(
    ::ICNF{<:AbstractFloat, <:ComputeMode, INPLACE, CONDITIONED, true},
    nn::LuxCore.AbstractLuxLayer,
    ::Number,
) where {INPLACE, CONDITIONED}
    return nn
end

function reg_z(
    ::ICNF{
        <:AbstractFloat,
        <:VectorMode,
        INPLACE,
        CONDITIONED,
        AUTONOMOUS,
        AUGMENTED,
        STEER,
        true,
    },
    ::TrainMode{true},
    ż::Any,
) where {INPLACE, CONDITIONED, AUTONOMOUS, AUGMENTED, STEER}
    return LinearAlgebra.norm(ż)
end

function reg_z(::ICNF{T, <:VectorMode}, ::Mode, ::Any) where {T <: AbstractFloat}
    return zero(T)
end

function reg_z(
    ::ICNF{
        <:AbstractFloat,
        <:MatrixMode,
        INPLACE,
        CONDITIONED,
        AUTONOMOUS,
        AUGMENTED,
        STEER,
        true,
    },
    ::TrainMode{true},
    ż::Any,
) where {INPLACE, CONDITIONED, AUTONOMOUS, AUGMENTED, STEER}
    return LinearAlgebra.norm.(eachcol(ż))
end

function reg_z(::ICNF{T, <:MatrixMode}, ::Mode, ż::Any) where {T <: AbstractFloat}
    zrs_Ė = similar(ż, size(ż, 2))
    ChainRulesCore.@ignore_derivatives fill!(zrs_Ė, zero(T))
    return zrs_Ė
end

function reg_j(
    ::ICNF{
        <:AbstractFloat,
        <:VectorMode,
        INPLACE,
        CONDITIONED,
        AUTONOMOUS,
        AUGMENTED,
        STEER,
        NORM_Z,
        true,
    },
    ::TrainMode{true},
    ϵ_J::Any,
) where {INPLACE, CONDITIONED, AUTONOMOUS, AUGMENTED, STEER, NORM_Z}
    return LinearAlgebra.norm(ϵ_J)
end

function reg_j(::ICNF{T, <:VectorMode}, ::Mode, ::Any) where {T <: AbstractFloat}
    return zero(T)
end

function reg_j(
    ::ICNF{
        <:AbstractFloat,
        <:MatrixMode,
        INPLACE,
        CONDITIONED,
        AUTONOMOUS,
        AUGMENTED,
        STEER,
        NORM_Z,
        true,
    },
    ::TrainMode{true},
    ϵ_J::Any,
) where {INPLACE, CONDITIONED, AUTONOMOUS, AUGMENTED, STEER, NORM_Z}
    return LinearAlgebra.norm.(eachcol(ϵ_J))
end

function reg_j(::ICNF{T, <:MatrixMode}, ::Mode, ϵ_J::Any) where {T <: AbstractFloat}
    zrs_ṅ = similar(ϵ_J, size(ϵ_J, 2))
    ChainRulesCore.@ignore_derivatives fill!(zrs_ṅ, zero(T))
    return zrs_ṅ
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::ICNF{T, <:DIVectorMode, false},
    mode::TestMode,
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat}
    n_aug = n_augments(icnf, mode)
    nn = add_time_nn(icnf, nn, t)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1)]
    ż, J = icnf_jacobian(icnf, mode, snn, z)
    l̇ = -LinearAlgebra.tr(J)
    Ė = reg_z(icnf, mode, ż)
    ṅ = reg_j(icnf, mode, ż)
    return vcat(ż, l̇, Ė, ṅ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::ICNF{T, <:DIVectorMode, true},
    mode::TestMode,
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat}
    n_aug = n_augments(icnf, mode)
    nn = add_time_nn(icnf, nn, t)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1)]
    ż, J = icnf_jacobian(icnf, mode, snn, z)
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -LinearAlgebra.tr(J)
    du[(end - n_aug + 1)] = reg_z(icnf, mode, ż)
    du[(end - n_aug + 2)] = reg_j(icnf, mode, ż)
    return nothing
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::ICNF{T, <:MatrixMode, false},
    mode::TestMode,
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat}
    n_aug = n_augments(icnf, mode)
    nn = add_time_nn(icnf, nn, t)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, J = icnf_jacobian(icnf, mode, snn, z)
    l̇ = -transpose(LinearAlgebra.tr.(eachslice(J; dims = 3)))
    Ė = transpose(reg_z(icnf, mode, ż))
    ṅ = transpose(reg_j(icnf, mode, ż))
    return vcat(ż, l̇, Ė, ṅ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::ICNF{T, <:MatrixMode, true},
    mode::TestMode,
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat}
    n_aug = n_augments(icnf, mode)
    nn = add_time_nn(icnf, nn, t)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, J = icnf_jacobian(icnf, mode, snn, z)
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -(LinearAlgebra.tr.(eachslice(J; dims = 3)))
    du[(end - n_aug + 1), :] .= reg_z(icnf, mode, ż)
    du[(end - n_aug + 2), :] .= reg_j(icnf, mode, ż)
    return nothing
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::ICNF{T, <:DIVecJacVectorMode, false},
    mode::TrainMode,
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat}
    n_aug = n_augments(icnf, mode)
    nn = add_time_nn(icnf, nn, t)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1)]
    ż, ϵJ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    l̇ = -LinearAlgebra.dot(ϵJ, ϵ)
    Ė = reg_z(icnf, mode, ż)
    ṅ = reg_j(icnf, mode, ϵJ)
    return vcat(ż, l̇, Ė, ṅ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::ICNF{T, <:DIVecJacVectorMode, true},
    mode::TrainMode,
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat}
    n_aug = n_augments(icnf, mode)
    nn = add_time_nn(icnf, nn, t)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1)]
    ż, ϵJ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -LinearAlgebra.dot(ϵJ, ϵ)
    du[(end - n_aug + 1)] = reg_z(icnf, mode, ż)
    du[(end - n_aug + 2)] = reg_j(icnf, mode, ϵJ)
    return nothing
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::ICNF{T, <:DIJacVecVectorMode, false},
    mode::TrainMode,
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat}
    n_aug = n_augments(icnf, mode)
    nn = add_time_nn(icnf, nn, t)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1)]
    ż, Jϵ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    l̇ = -LinearAlgebra.dot(ϵ, Jϵ)
    Ė = reg_z(icnf, mode, ż)
    ṅ = reg_j(icnf, mode, Jϵ)
    return vcat(ż, l̇, Ė, ṅ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::ICNF{T, <:DIJacVecVectorMode, true},
    mode::TrainMode,
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractVector{T},
) where {T <: AbstractFloat}
    n_aug = n_augments(icnf, mode)
    nn = add_time_nn(icnf, nn, t)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1)]
    ż, Jϵ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -LinearAlgebra.dot(ϵ, Jϵ)
    du[(end - n_aug + 1)] = reg_z(icnf, mode, ż)
    du[(end - n_aug + 2)] = reg_j(icnf, mode, Jϵ)
    return nothing
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::ICNF{T, <:DIVecJacMatrixMode, false},
    mode::TrainMode,
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat}
    n_aug = n_augments(icnf, mode)
    nn = add_time_nn(icnf, nn, t)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, ϵJ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    l̇ = -sum(ϵJ .* ϵ; dims = 1)
    Ė = transpose(reg_z(icnf, mode, ż))
    ṅ = transpose(reg_j(icnf, mode, ϵJ))
    return vcat(ż, l̇, Ė, ṅ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::ICNF{T, <:DIVecJacMatrixMode, true},
    mode::TrainMode,
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat}
    n_aug = n_augments(icnf, mode)
    nn = add_time_nn(icnf, nn, t)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, ϵJ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -vec(sum(ϵJ .* ϵ; dims = 1))
    du[(end - n_aug + 1), :] .= reg_z(icnf, mode, ż)
    du[(end - n_aug + 2), :] .= reg_j(icnf, mode, ϵJ)
    return nothing
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::ICNF{T, <:DIJacVecMatrixMode, false},
    mode::TrainMode,
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat}
    n_aug = n_augments(icnf, mode)
    nn = add_time_nn(icnf, nn, t)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, Jϵ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    l̇ = -sum(ϵ .* Jϵ; dims = 1)
    Ė = transpose(reg_z(icnf, mode, ż))
    ṅ = transpose(reg_j(icnf, mode, Jϵ))
    return vcat(ż, l̇, Ė, ṅ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::ICNF{T, <:DIJacVecMatrixMode, true},
    mode::TrainMode,
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat}
    n_aug = n_augments(icnf, mode)
    nn = add_time_nn(icnf, nn, t)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, Jϵ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -vec(sum(ϵ .* Jϵ; dims = 1))
    du[(end - n_aug + 1), :] .= reg_z(icnf, mode, ż)
    du[(end - n_aug + 2), :] .= reg_j(icnf, mode, Jϵ)
    return nothing
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::ICNF{T, <:LuxVecJacMatrixMode, false},
    mode::TrainMode,
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat}
    n_aug = n_augments(icnf, mode)
    nn = add_time_nn(icnf, nn, t)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, ϵJ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    l̇ = -sum(ϵJ .* ϵ; dims = 1)
    Ė = transpose(reg_z(icnf, mode, ż))
    ṅ = transpose(reg_j(icnf, mode, ϵJ))
    return vcat(ż, l̇, Ė, ṅ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::ICNF{T, <:LuxVecJacMatrixMode, true},
    mode::TrainMode,
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat}
    n_aug = n_augments(icnf, mode)
    nn = add_time_nn(icnf, nn, t)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, ϵJ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -vec(sum(ϵJ .* ϵ; dims = 1))
    du[(end - n_aug + 1), :] .= reg_z(icnf, mode, ż)
    du[(end - n_aug + 2), :] .= reg_j(icnf, mode, ϵJ)
    return nothing
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::ICNF{T, <:LuxJacVecMatrixMode, false},
    mode::TrainMode,
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat}
    n_aug = n_augments(icnf, mode)
    nn = add_time_nn(icnf, nn, t)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, Jϵ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    l̇ = -sum(ϵ .* Jϵ; dims = 1)
    Ė = transpose(reg_z(icnf, mode, ż))
    ṅ = transpose(reg_j(icnf, mode, Jϵ))
    return vcat(ż, l̇, Ė, ṅ)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::ICNF{T, <:LuxJacVecMatrixMode, true},
    mode::TrainMode,
    nn::LuxCore.AbstractLuxLayer,
    st::NamedTuple,
    ϵ::AbstractMatrix{T},
) where {T <: AbstractFloat}
    n_aug = n_augments(icnf, mode)
    nn = add_time_nn(icnf, nn, t)
    snn = LuxCore.StatefulLuxLayer{true}(nn, p, st)
    z = u[begin:(end - n_aug - 1), :]
    ż, Jϵ = icnf_jacobian(icnf, mode, snn, z, ϵ)
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -vec(sum(ϵ .* Jϵ; dims = 1))
    du[(end - n_aug + 1), :] .= reg_z(icnf, mode, ż)
    du[(end - n_aug + 2), :] .= reg_j(icnf, mode, Jϵ)
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

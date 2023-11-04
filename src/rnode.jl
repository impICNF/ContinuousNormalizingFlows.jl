export RNODE

"""
Implementation of RNODE from

[Finlay, Chris, Jörn-Henrik Jacobsen, Levon Nurbekyan, and Adam M. Oberman. "How to train your neural ODE: the world of Jacobian and kinetic regularization." arXiv preprint arXiv:2002.02798 (2020).](https://arxiv.org/abs/2002.02798)
"""
struct RNODE{
    T <: AbstractFloat,
    CM <: ComputeMode,
    INPLACE,
    AUGMENTED,
    STEER,
    NN <: LuxCore.AbstractExplicitLayer,
    NVARS <: Int,
    RESOURCE <: AbstractResource,
    BASEDIST <: Distribution,
    TSPAN <: NTuple{2, T},
    STEERDIST <: Distribution,
    DIFFERENTIATION_BACKEND <: AbstractDifferentiation.AbstractBackend,
    AUTODIFF_BACKEND <: ADTypes.AbstractADType,
    SOL_KWARGS <: NamedTuple,
    RNG <: AbstractRNG,
} <: AbstractICNF{T, CM, INPLACE, AUGMENTED, STEER}
    nn::NN
    nvars::NVARS
    naugmented::NVARS

    resource::RESOURCE
    basedist::BASEDIST
    tspan::TSPAN
    steerdist::STEERDIST
    differentiation_backend::DIFFERENTIATION_BACKEND
    autodiff_backend::AUTODIFF_BACKEND
    sol_kwargs::SOL_KWARGS
    rng::RNG
    λ₁::T
    λ₂::T
end

function construct(
    aicnf::Type{<:RNODE},
    nn,
    nvars::Int,
    naugmented::Int = 0;
    data_type::Type{<:AbstractFloat} = Float32,
    compute_mode::Type{<:ComputeMode} = ADVecJacVectorMode,
    inplace::Bool = false,
    resource::AbstractResource = CPU1(),
    basedist::Distribution = MvNormal(
        Zeros{data_type}(nvars + naugmented),
        Eye{data_type}(nvars + naugmented),
    ),
    tspan::NTuple{2} = (zero(data_type), one(data_type)),
    steer_rate::AbstractFloat = zero(data_type),
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    autodiff_backend::ADTypes.AbstractADType = ifelse(
        compute_mode <: SDJacVecMatrixMode,
        AutoForwardDiff(),
        AutoZygote(),
    ),
    sol_kwargs::NamedTuple = (
        alg_hints = [:nonstiff, :memorybound],
        save_everystep = false,
        alg = VCABM(),
        sensealg = InterpolatingAdjoint(;
            autodiff = true,
            autojacvec = ZygoteVJP(),
            checkpointing = true,
        ),
        reltol = sqrt(eps(one(Float32))),
        abstol = eps(one(Float32)),
        maxiters = typemax(Int32),
    ),
    rng::AbstractRNG = Random.default_rng(),
    λ₁::AbstractFloat = convert(data_type, 1e-2),
    λ₂::AbstractFloat = convert(data_type, 1e-2),
)
    steerdist = Uniform{data_type}(-steer_rate, steer_rate)

    aicnf{
        data_type,
        compute_mode,
        inplace,
        !iszero(naugmented),
        !iszero(steer_rate),
        typeof(nn),
        typeof(nvars),
        typeof(resource),
        typeof(basedist),
        typeof(tspan),
        typeof(steerdist),
        typeof(differentiation_backend),
        typeof(autodiff_backend),
        typeof(sol_kwargs),
        typeof(rng),
    }(
        nn,
        nvars,
        naugmented,
        resource,
        basedist,
        tspan,
        steerdist,
        differentiation_backend,
        autodiff_backend,
        sol_kwargs,
        rng,
        λ₁,
        λ₂,
    )
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::RNODE{T, <:ADVecJacVectorMode},
    mode::TrainMode,
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, VJ = AbstractDifferentiation.value_and_pullback_function(
        icnf.differentiation_backend,
        let p = p, st = st
            x -> first(icnf.nn(x, p, st))
        end,
        z,
    )
    ϵJ = only(VJ(ϵ))
    l̇ = ϵJ ⋅ ϵ
    Ė = norm(ż)
    ṅ = norm(ϵJ)
    vcat(ż, -l̇, Ė, ṅ)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::RNODE{T, <:ADJacVecVectorMode},
    mode::TrainMode,
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż_JV = AbstractDifferentiation.value_and_pushforward_function(
        icnf.differentiation_backend,
        let p = p, st = st
            x -> first(icnf.nn(x, p, st))
        end,
        z,
    )
    ż, Jϵ = ż_JV(ϵ)
    Jϵ = only(Jϵ)
    l̇ = ϵ ⋅ Jϵ
    Ė = norm(ż)
    ṅ = norm(Jϵ)
    vcat(ż, -l̇, Ė, ṅ)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::RNODE{T, <:ZygoteVectorMode},
    mode::TrainMode,
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, VJ = Zygote.pullback(let p = p, st = st
        x -> first(icnf.nn(x, p, st))
    end, z)
    ϵJ = only(VJ(ϵ))
    l̇ = ϵJ ⋅ ϵ
    Ė = norm(ż)
    ṅ = norm(ϵJ)
    vcat(ż, -l̇, Ė, ṅ)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::RNODE{T, <:SDVecJacMatrixMode},
    mode::TrainMode,
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż = first(icnf.nn(z, p, st))
    Jf = VecJac(let p = p, st = st
        x -> first(icnf.nn(x, p, st))
    end, z; autodiff = icnf.autodiff_backend)
    ϵJ = reshape(Jf * ϵ, size(z))
    l̇ = sum(ϵJ .* ϵ; dims = 1)
    Ė = transpose(norm.(eachcol(ż)))
    ṅ = transpose(norm.(eachcol(ϵJ)))
    vcat(ż, -l̇, Ė, ṅ)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::RNODE{T, <:SDJacVecMatrixMode},
    mode::TrainMode,
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż = first(icnf.nn(z, p, st))
    Jf = JacVec(let p = p, st = st
        x -> first(icnf.nn(x, p, st))
    end, z; autodiff = icnf.autodiff_backend)
    Jϵ = reshape(Jf * ϵ, size(z))
    l̇ = sum(ϵ .* Jϵ; dims = 1)
    Ė = transpose(norm.(eachcol(ż)))
    ṅ = transpose(norm.(eachcol(Jϵ)))
    vcat(ż, -l̇, Ė, ṅ)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::RNODE{T, <:ZygoteMatrixMode},
    mode::TrainMode,
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż, VJ = Zygote.pullback(let p = p, st = st
        x -> first(icnf.nn(x, p, st))
    end, z)
    ϵJ = only(VJ(ϵ))
    l̇ = sum(ϵJ .* ϵ; dims = 1)
    Ė = transpose(norm.(eachcol(ż)))
    ṅ = transpose(norm.(eachcol(ϵJ)))
    vcat(ż, -l̇, Ė, ṅ)
end

@views function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::RNODE{T, <:ZygoteMatrixModeInplace, true},
    mode::TrainMode,
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż, VJ = Zygote.pullback(let p = p, st = st
        x -> first(icnf.nn(x, p, st))
    end, z)
    ϵJ = only(VJ(ϵ))
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -vec(sum(ϵJ .* ϵ; dims = 1))
    du[(end - n_aug + 1), :] .= norm.(eachcol(ż))
    du[(end - n_aug + 2), :] .= norm.(eachcol(ϵJ))
    nothing
end

@inline function loss(
    icnf::RNODE{<:AbstractFloat, <:VectorMode},
    mode::TrainMode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::Any,
)
    logp̂x, Ė, ṅ = inference(icnf, mode, xs, ps, st)
    -logp̂x + icnf.λ₁ * Ė + icnf.λ₂ * ṅ
end

@inline function loss(
    icnf::RNODE{<:AbstractFloat, <:MatrixMode},
    mode::TrainMode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::Any,
)
    logp̂x, Ė, ṅ = inference(icnf, mode, xs, ps, st)
    mean(-logp̂x + icnf.λ₁ * Ė + icnf.λ₂ * ṅ)
end

@inline function n_augment(::RNODE, ::TrainMode)
    2
end

export CondFFJORD

"""
Implementation of FFJORD (Conditional Version)
"""
struct CondFFJORD{
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
} <: AbstractCondICNF{T, CM, INPLACE, AUGMENTED, STEER}
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
end

export CondRNODE

"""
Implementation of RNODE (Conditional Version)
"""
struct CondRNODE{
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
} <: AbstractCondICNF{T, CM, INPLACE, AUGMENTED, STEER}
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

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:ADVectorMode, false},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, J = AbstractDifferentiation.value_and_jacobian(
        icnf.differentiation_backend,
        let ys = ys, p = p, st = st
            x -> first(icnf.nn(vcat(x, ys), p, st))
        end,
        z,
    )
    l̇ = -tr(only(J))
    vcat(ż, l̇)
end

@views function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:ADVectorMode, true},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, J = AbstractDifferentiation.value_and_jacobian(
        icnf.differentiation_backend,
        let ys = ys, p = p, st = st
            x -> first(icnf.nn(vcat(x, ys), p, st))
        end,
        z,
    )
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -tr(only(J))
    nothing
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:ZygoteVectorMode, false},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, J = Zygote.withjacobian(let ys = ys, p = p, st = st
        x -> first(icnf.nn(vcat(x, ys), p, st))
    end, z)
    l̇ = -tr(only(J))
    vcat(ż, l̇)
end

@views function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:ZygoteVectorMode, true},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, J = Zygote.withjacobian(let ys = ys, p = p, st = st
        x -> first(icnf.nn(vcat(x, ys), p, st))
    end, z)
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -tr(only(J))
    nothing
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:MatrixMode, false},
    mode::TestMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż, J = jacobian_batched(icnf, let ys = ys, p = p, st = st
        x -> first(icnf.nn(vcat(x, ys), p, st))
    end, z)
    l̇ = -transpose(tr.(eachslice(J; dims = 3)))
    vcat(ż, l̇)
end

@views function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:MatrixMode, true},
    mode::TestMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż, J = jacobian_batched(icnf, let ys = ys, p = p, st = st
        x -> first(icnf.nn(vcat(x, ys), p, st))
    end, z)
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -(tr.(eachslice(J; dims = 3)))
    nothing
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:ADVecJacVectorMode, false},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, VJ = AbstractDifferentiation.value_and_pullback_function(
        icnf.differentiation_backend,
        let ys = ys, p = p, st = st
            x -> first(icnf.nn(vcat(x, ys), p, st))
        end,
        z,
    )
    ϵJ = convert(AbstractArray{<:Real}, only(VJ(ϵ)))
    l̇ = -(ϵJ ⋅ ϵ)
    if icnf isa CondRNODE
        Ė = norm(ż)
        ṅ = norm(ϵJ)
        vcat(ż, l̇, Ė, ṅ)
    else
        vcat(ż, l̇)
    end
end

@views function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:ADVecJacVectorMode, true},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, VJ = AbstractDifferentiation.value_and_pullback_function(
        icnf.differentiation_backend,
        let ys = ys, p = p, st = st
            x -> first(icnf.nn(vcat(x, ys), p, st))
        end,
        z,
    )
    ϵJ = convert(AbstractArray{<:Real}, only(VJ(ϵ)))
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -(ϵJ ⋅ ϵ)
    if icnf isa CondRNODE
        du[(end - n_aug + 1)] = norm(ż)
        du[(end - n_aug + 2)] = norm(ϵJ)
    end
    nothing
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:ADJacVecVectorMode, false},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż_JV = AbstractDifferentiation.value_and_pushforward_function(
        icnf.differentiation_backend,
        let ys = ys, p = p, st = st
            x -> first(icnf.nn(vcat(x, ys), p, st))
        end,
        z,
    )
    ż, Jϵ = ż_JV(ϵ)
    Jϵ = convert(AbstractArray{<:Real}, only(Jϵ))
    l̇ = -(ϵ ⋅ Jϵ)
    if icnf isa CondRNODE
        Ė = norm(ż)
        ṅ = norm(Jϵ)
        vcat(ż, l̇, Ė, ṅ)
    else
        vcat(ż, l̇)
    end
end

@views function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:ADJacVecVectorMode, true},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż_JV = AbstractDifferentiation.value_and_pushforward_function(
        icnf.differentiation_backend,
        let ys = ys, p = p, st = st
            x -> first(icnf.nn(vcat(x, ys), p, st))
        end,
        z,
    )
    ż, Jϵ = ż_JV(ϵ)
    Jϵ = convert(AbstractArray{<:Real}, only(Jϵ))
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -(ϵ ⋅ Jϵ)
    if icnf isa CondRNODE
        du[(end - n_aug + 1)] = norm(ż)
        du[(end - n_aug + 2)] = norm(Jϵ)
    end
    nothing
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:ZygoteVectorMode, false},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, VJ = Zygote.pullback(let ys = ys, p = p, st = st
        x -> first(icnf.nn(vcat(x, ys), p, st))
    end, z)
    ϵJ = convert(AbstractArray{<:Real}, only(VJ(ϵ)))
    l̇ = -(ϵJ ⋅ ϵ)
    if icnf isa CondRNODE
        Ė = norm(ż)
        ṅ = norm(ϵJ)
        vcat(ż, l̇, Ė, ṅ)
    else
        vcat(ż, l̇)
    end
end

@views function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:ZygoteVectorMode, true},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, VJ = Zygote.pullback(let ys = ys, p = p, st = st
        x -> first(icnf.nn(vcat(x, ys), p, st))
    end, z)
    ϵJ = convert(AbstractArray{<:Real}, only(VJ(ϵ)))
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -(ϵJ ⋅ ϵ)
    if icnf isa CondRNODE
        du[(end - n_aug + 1)] = norm(ż)
        du[(end - n_aug + 2)] = norm(ϵJ)
    end
    nothing
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:SDVecJacMatrixMode, false},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż = first(icnf.nn(vcat(z, ys), p, st))
    Jf = VecJac(
        let ys = ys, p = p, st = st
            x -> first(icnf.nn(vcat(x, ys), p, st))
        end,
        z;
        autodiff = icnf.autodiff_backend,
    )
    ϵJ = convert(AbstractArray{<:Real}, reshape(Jf * ϵ, size(z)))
    l̇ = -sum(ϵJ .* ϵ; dims = 1)
    if icnf isa CondRNODE
        Ė = transpose(norm.(eachcol(ż)))
        ṅ = transpose(norm.(eachcol(ϵJ)))
        vcat(ż, l̇, Ė, ṅ)
    else
        vcat(ż, l̇)
    end
end

@views function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:SDVecJacMatrixMode, true},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż = first(icnf.nn(vcat(z, ys), p, st))
    Jf = VecJac(
        let ys = ys, p = p, st = st
            x -> first(icnf.nn(vcat(x, ys), p, st))
        end,
        z;
        autodiff = icnf.autodiff_backend,
    )
    ϵJ = convert(AbstractArray{<:Real}, reshape(Jf * ϵ, size(z)))
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -vec(sum(ϵJ .* ϵ; dims = 1))
    if icnf isa CondRNODE
        du[(end - n_aug + 1), :] .= norm.(eachcol(ż))
        du[(end - n_aug + 2), :] .= norm.(eachcol(ϵJ))
    end
    nothing
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:SDJacVecMatrixMode, false},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż = first(icnf.nn(vcat(z, ys), p, st))
    Jf = JacVec(
        let ys = ys, p = p, st = st
            x -> first(icnf.nn(vcat(x, ys), p, st))
        end,
        z;
        autodiff = icnf.autodiff_backend,
    )
    Jϵ = convert(AbstractArray{<:Real}, reshape(Jf * ϵ, size(z)))
    l̇ = -sum(ϵ .* Jϵ; dims = 1)
    if icnf isa CondRNODE
        Ė = transpose(norm.(eachcol(ż)))
        ṅ = transpose(norm.(eachcol(Jϵ)))
        vcat(ż, l̇, Ė, ṅ)
    else
        vcat(ż, l̇)
    end
end

@views function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:SDJacVecMatrixMode, true},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż = first(icnf.nn(vcat(z, ys), p, st))
    Jf = JacVec(
        let ys = ys, p = p, st = st
            x -> first(icnf.nn(vcat(x, ys), p, st))
        end,
        z;
        autodiff = icnf.autodiff_backend,
    )
    Jϵ = convert(AbstractArray{<:Real}, reshape(Jf * ϵ, size(z)))
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -vec(sum(ϵ .* Jϵ; dims = 1))
    if icnf isa CondRNODE
        du[(end - n_aug + 1), :] .= norm.(eachcol(ż))
        du[(end - n_aug + 2), :] .= norm.(eachcol(Jϵ))
    end
    nothing
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:ZygoteMatrixMode, false},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż, VJ = Zygote.pullback(let ys = ys, p = p, st = st
        x -> first(icnf.nn(vcat(x, ys), p, st))
    end, z)
    ϵJ = convert(AbstractArray{<:Real}, only(VJ(ϵ)))
    l̇ = -sum(ϵJ .* ϵ; dims = 1)
    if icnf isa CondRNODE
        Ė = transpose(norm.(eachcol(ż)))
        ṅ = transpose(norm.(eachcol(ϵJ)))
        vcat(ż, l̇, Ė, ṅ)
    else
        vcat(ż, l̇)
    end
end

@views function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{T, <:ZygoteMatrixMode, true},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż, VJ = Zygote.pullback(let ys = ys, p = p, st = st
        x -> first(icnf.nn(vcat(x, ys), p, st))
    end, z)
    ϵJ = convert(AbstractArray{<:Real}, only(VJ(ϵ)))
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -vec(sum(ϵJ .* ϵ; dims = 1))
    if icnf isa CondRNODE
        du[(end - n_aug + 1), :] .= norm.(eachcol(ż))
        du[(end - n_aug + 2), :] .= norm.(eachcol(ϵJ))
    end
    nothing
end

@inline function loss(
    icnf::CondRNODE{<:AbstractFloat, <:VectorMode},
    mode::TrainMode,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any,
)
    logp̂x, (Ė, ṅ) = inference(icnf, mode, xs, ys, ps, st)
    -logp̂x + icnf.λ₁ * Ė + icnf.λ₂ * ṅ
end

@inline function loss(
    icnf::CondRNODE{<:AbstractFloat, <:MatrixMode},
    mode::TrainMode,
    xs::AbstractMatrix{<:Real},
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::Any,
)
    logp̂x, (Ė, ṅ) = inference(icnf, mode, xs, ys, ps, st)
    mean(-logp̂x + icnf.λ₁ * Ė + icnf.λ₂ * ṅ)
end

@inline function n_augment(::CondRNODE, ::TrainMode)
    2
end

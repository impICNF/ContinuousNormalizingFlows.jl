export FFJORD

"""
Implementation of FFJORD from

[Grathwohl, Will, Ricky TQ Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. "Ffjord: Free-form continuous dynamics for scalable reversible generative models." arXiv preprint arXiv:1810.01367 (2018).](https://arxiv.org/abs/1810.01367)
"""
struct FFJORD{
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
end

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

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractICNF{T, <:ADVectorMode, false},
    mode::TestMode,
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, J = AbstractDifferentiation.value_and_jacobian(
        icnf.differentiation_backend,
        let p = p, st = st
            x -> first(icnf.nn(x, p, st))
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
    icnf::AbstractICNF{T, <:ADVectorMode, true},
    mode::TestMode,
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, J = AbstractDifferentiation.value_and_jacobian(
        icnf.differentiation_backend,
        let p = p, st = st
            x -> first(icnf.nn(x, p, st))
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
    icnf::AbstractICNF{T, <:ZygoteVectorMode, false},
    mode::TestMode,
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, J = Zygote.withjacobian(let p = p, st = st
        x -> first(icnf.nn(x, p, st))
    end, z)
    l̇ = -tr(only(J))
    vcat(ż, l̇)
end

@views function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractICNF{T, <:ZygoteVectorMode, true},
    mode::TestMode,
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, J = Zygote.withjacobian(let p = p, st = st
        x -> first(icnf.nn(x, p, st))
    end, z)
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -tr(only(J))
    nothing
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractICNF{T, <:MatrixMode, false},
    mode::TestMode,
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż, J = jacobian_batched(icnf, let p = p, st = st
        x -> first(icnf.nn(x, p, st))
    end, z)
    l̇ = -transpose(tr.(eachslice(J; dims = 3)))
    vcat(ż, l̇)
end

@views function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractICNF{T, <:MatrixMode, true},
    mode::TestMode,
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż, J = jacobian_batched(icnf, let p = p, st = st
        x -> first(icnf.nn(x, p, st))
    end, z)
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -(tr.(eachslice(J; dims = 3)))
    nothing
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractICNF{T, <:ADVecJacVectorMode, false},
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
    ϵJ = oftype(z, only(VJ(ϵ)))
    l̇ = -(ϵJ ⋅ ϵ)
    if icnf isa RNODE
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
    icnf::AbstractICNF{T, <:ADVecJacVectorMode, true},
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
    ϵJ = oftype(z, only(VJ(ϵ)))
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -(ϵJ ⋅ ϵ)
    if icnf isa RNODE
        du[(end - n_aug + 1)] = norm(ż)
        du[(end - n_aug + 2)] = norm(ϵJ)
    end
    nothing
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractICNF{T, <:ADJacVecVectorMode, false},
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
    Jϵ = oftype(z, only(Jϵ))
    l̇ = -(ϵ ⋅ Jϵ)
    if icnf isa RNODE
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
    icnf::AbstractICNF{T, <:ADJacVecVectorMode, true},
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
    Jϵ = oftype(z, only(Jϵ))
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -(ϵ ⋅ Jϵ)
    if icnf isa RNODE
        du[(end - n_aug + 1)] = norm(ż)
        du[(end - n_aug + 2)] = norm(Jϵ)
    end
    nothing
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractICNF{T, <:ZygoteVectorMode, false},
    mode::TrainMode,
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, VJ = Zygote.pullback(let p = p, st = st
        x -> first(icnf.nn(x, p, st))
    end, z)
    ϵJ = oftype(z, only(VJ(ϵ)))
    l̇ = -(ϵJ ⋅ ϵ)
    if icnf isa RNODE
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
    icnf::AbstractICNF{T, <:ZygoteVectorMode, true},
    mode::TrainMode,
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż, VJ = Zygote.pullback(let p = p, st = st
        x -> first(icnf.nn(x, p, st))
    end, z)
    ϵJ = oftype(z, only(VJ(ϵ)))
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -(ϵJ ⋅ ϵ)
    if icnf isa RNODE
        du[(end - n_aug + 1)] = norm(ż)
        du[(end - n_aug + 2)] = norm(ϵJ)
    end
    nothing
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractICNF{T, <:SDVecJacMatrixMode, false},
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
    ϵJ = oftype(z, reshape(Jf * ϵ, size(z)))
    l̇ = -sum(ϵJ .* ϵ; dims = 1)
    if icnf isa RNODE
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
    icnf::AbstractICNF{T, <:SDVecJacMatrixMode, true},
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
    ϵJ = oftype(z, reshape(Jf * ϵ, size(z)))
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -vec(sum(ϵJ .* ϵ; dims = 1))
    if icnf isa RNODE
        du[(end - n_aug + 1), :] .= norm.(eachcol(ż))
        du[(end - n_aug + 2), :] .= norm.(eachcol(ϵJ))
    end
    nothing
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractICNF{T, <:SDJacVecMatrixMode, false},
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
    Jϵ = oftype(z, reshape(Jf * ϵ, size(z)))
    l̇ = -sum(ϵ .* Jϵ; dims = 1)
    if icnf isa RNODE
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
    icnf::AbstractICNF{T, <:SDJacVecMatrixMode, true},
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
    Jϵ = oftype(z, reshape(Jf * ϵ, size(z)))
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -vec(sum(ϵ .* Jϵ; dims = 1))
    if icnf isa RNODE
        du[(end - n_aug + 1), :] .= norm.(eachcol(ż))
        du[(end - n_aug + 2), :] .= norm.(eachcol(Jϵ))
    end
    nothing
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractICNF{T, <:ZygoteMatrixMode, false},
    mode::TrainMode,
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż, VJ = Zygote.pullback(let p = p, st = st
        x -> first(icnf.nn(x, p, st))
    end, z)
    ϵJ = oftype(z, only(VJ(ϵ)))
    l̇ = -sum(ϵJ .* ϵ; dims = 1)
    if icnf isa RNODE
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
    icnf::AbstractICNF{T, <:ZygoteMatrixMode, true},
    mode::TrainMode,
    ϵ::AbstractMatrix{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż, VJ = Zygote.pullback(let p = p, st = st
        x -> first(icnf.nn(x, p, st))
    end, z)
    ϵJ = oftype(z, only(VJ(ϵ)))
    du[begin:(end - n_aug - 1), :] .= ż
    du[(end - n_aug), :] .= -vec(sum(ϵJ .* ϵ; dims = 1))
    if icnf isa RNODE
        du[(end - n_aug + 1), :] .= norm.(eachcol(ż))
        du[(end - n_aug + 2), :] .= norm.(eachcol(ϵJ))
    end
    nothing
end

@inline function loss(
    icnf::RNODE{<:AbstractFloat, <:VectorMode},
    mode::TrainMode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::Any,
)
    logp̂x, (Ė, ṅ) = inference(icnf, mode, xs, ps, st)
    -logp̂x + icnf.λ₁ * Ė + icnf.λ₂ * ṅ
end

@inline function loss(
    icnf::RNODE{<:AbstractFloat, <:MatrixMode},
    mode::TrainMode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::Any,
)
    logp̂x, (Ė, ṅ) = inference(icnf, mode, xs, ps, st)
    mean(-logp̂x + icnf.λ₁ * Ė + icnf.λ₂ * ṅ)
end

@inline function n_augment(::RNODE, ::TrainMode)
    2
end

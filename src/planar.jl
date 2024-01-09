export Planar

"""
Implementation of Planar Flows from

[Chen, Ricky TQ, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. "Neural Ordinary Differential Equations." arXiv preprint arXiv:1806.07366 (2018).](https://arxiv.org/abs/1806.07366)
"""
struct Planar{
    T <: AbstractFloat,
    CM <: ComputeMode,
    INPLACE,
    AUGMENTED,
    STEER,
    NN <: PlanarLayer,
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

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::Planar{T, <:ADVectorMode},
    mode::TestMode,
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż = first(icnf.nn(z, p, st))
    l̇ = -(
        p.u ⋅ transpose(
            only(
                AbstractDifferentiation.jacobian(
                    icnf.differentiation_backend,
                    let p = p, st = st
                        x -> first(pl_h(icnf.nn, x, p, st))
                    end,
                    z,
                ),
            ),
        )
    )
    vcat(ż, l̇)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::Planar{T, <:ADVectorMode},
    mode::TrainMode,
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż = first(icnf.nn(z, p, st))
    l̇ = -(
        p.u ⋅ transpose(
            only(
                AbstractDifferentiation.jacobian(
                    icnf.differentiation_backend,
                    let p = p, st = st
                        x -> first(pl_h(icnf.nn, x, p, st))
                    end,
                    z,
                ),
            ),
        )
    )
    vcat(ż, l̇)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::Planar{T, <:ZygoteVectorMode},
    mode::TestMode,
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż = first(icnf.nn(z, p, st))
    l̇ = -(
        p.u ⋅ transpose(only(Zygote.jacobian(let p = p, st = st
            x -> first(pl_h(icnf.nn, x, p, st))
        end, z)))
    )
    vcat(ż, l̇)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::Planar{T, <:ZygoteVectorMode},
    mode::TrainMode,
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż = first(icnf.nn(z, p, st))
    l̇ = -(
        p.u ⋅ transpose(only(Zygote.jacobian(let p = p, st = st
            x -> first(pl_h(icnf.nn, x, p, st))
        end, z)))
    )
    vcat(ż, l̇)
end

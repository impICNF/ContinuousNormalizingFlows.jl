export CondPlanar

"""
Implementation of Planar (Conditional Version)
"""
struct CondPlanar{
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
    EPSDIST <: Distribution,
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
    epsdist::EPSDIST
    differentiation_backend::DIFFERENTIATION_BACKEND
    autodiff_backend::AUTODIFF_BACKEND
    sol_kwargs::SOL_KWARGS
    rng::RNG
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{T, <:ADVectorMode, false},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż = first(icnf.nn(vcat(z, ys), p, st))
    l̇ = -(
        p.u ⋅ transpose(
            only(
                AbstractDifferentiation.jacobian(
                    icnf.differentiation_backend,
                    let ys = ys, p = p, st = st
                        x -> first(pl_h(icnf.nn, vcat(x, ys), p, st))
                    end,
                    z,
                ),
            ),
        )
    )
    vcat(ż, l̇)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{T, <:ADVectorMode, true},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż = first(icnf.nn(vcat(z, ys), p, st))
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -(
        p.u ⋅ transpose(
            only(
                AbstractDifferentiation.jacobian(
                    icnf.differentiation_backend,
                    let ys = ys, p = p, st = st
                        x -> first(pl_h(icnf.nn, vcat(x, ys), p, st))
                    end,
                    z,
                ),
            ),
        )
    )
    nothing
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{T, <:ADVectorMode, false},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż = first(icnf.nn(vcat(z, ys), p, st))
    l̇ = -(
        p.u ⋅ transpose(
            only(
                AbstractDifferentiation.jacobian(
                    icnf.differentiation_backend,
                    let ys = ys, p = p, st = st
                        x -> first(pl_h(icnf.nn, vcat(x, ys), p, st))
                    end,
                    z,
                ),
            ),
        )
    )
    vcat(ż, l̇)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{T, <:ADVectorMode, true},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż = first(icnf.nn(vcat(z, ys), p, st))
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -(
        p.u ⋅ transpose(
            only(
                AbstractDifferentiation.jacobian(
                    icnf.differentiation_backend,
                    let ys = ys, p = p, st = st
                        x -> first(pl_h(icnf.nn, vcat(x, ys), p, st))
                    end,
                    z,
                ),
            ),
        )
    )
    nothing
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{T, <:ZygoteVectorMode, false},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż = first(icnf.nn(vcat(z, ys), p, st))
    l̇ = -(
        p.u ⋅ transpose(
            only(Zygote.jacobian(let ys = ys, p = p, st = st
                x -> first(pl_h(icnf.nn, vcat(x, ys), p, st))
            end, z)),
        )
    )
    vcat(ż, l̇)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{T, <:ZygoteVectorMode, true},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż = first(icnf.nn(vcat(z, ys), p, st))
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -(
        p.u ⋅ transpose(
            only(Zygote.jacobian(let ys = ys, p = p, st = st
                x -> first(pl_h(icnf.nn, vcat(x, ys), p, st))
            end, z)),
        )
    )
    nothing
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{T, <:ZygoteVectorMode, false},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż = first(icnf.nn(vcat(z, ys), p, st))
    l̇ = -(
        p.u ⋅ transpose(
            only(Zygote.jacobian(let ys = ys, p = p, st = st
                x -> first(pl_h(icnf.nn, vcat(x, ys), p, st))
            end, z)),
        )
    )
    vcat(ż, l̇)
end

function augmented_f(
    du::Any,
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{T, <:ZygoteVectorMode, true},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    ż = first(icnf.nn(vcat(z, ys), p, st))
    du[begin:(end - n_aug - 1)] .= ż
    du[(end - n_aug)] = -(
        p.u ⋅ transpose(
            only(Zygote.jacobian(let ys = ys, p = p, st = st
                x -> first(pl_h(icnf.nn, vcat(x, ys), p, st))
            end, z)),
        )
    )
    nothing
end

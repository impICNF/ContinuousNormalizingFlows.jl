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

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{T, <:ADVectorMode},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    mz = first(icnf.nn(vcat(z, ys), p, st))
    trace_J =
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
    vcat(mz, -trace_J)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{T, <:ADVectorMode},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    mz = first(icnf.nn(vcat(z, ys), p, st))
    trace_J =
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
    vcat(mz, -trace_J)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{T, <:ZygoteVectorMode},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    mz = first(icnf.nn(vcat(z, ys), p, st))
    trace_J =
        p.u ⋅ transpose(
            only(Zygote.jacobian(let ys = ys, p = p, st = st
                x -> first(pl_h(icnf.nn, vcat(x, ys), p, st))
            end, z)),
        )
    vcat(mz, -trace_J)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondPlanar{T, <:ZygoteVectorMode},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{T},
    st::Any,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    mz = first(icnf.nn(vcat(z, ys), p, st))
    trace_J =
        p.u ⋅ transpose(
            only(Zygote.jacobian(let ys = ys, p = p, st = st
                x -> first(pl_h(icnf.nn, vcat(x, ys), p, st))
            end, z)),
        )
    vcat(mz, -trace_J)
end

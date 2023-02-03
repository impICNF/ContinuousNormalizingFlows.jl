export CondPlanar

"""
Implementation of Planar (Conditional Version)
"""
struct CondPlanar{T <: AbstractFloat, AT <: AbstractArray} <: AbstractCondICNF{T, AT}
    nn::PlanarLayer

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    differentiation_backend::AbstractDifferentiation.AbstractBackend

    # trace_test
    # trace_train
end

function CondPlanar{T, AT}(
    nn::PlanarLayer,
    nvars::Integer,
    ;
    basedist::Distribution = MvNormal(Zeros{T}(nvars), one(T) * I),
    tspan::Tuple{T, T} = convert(Tuple{T, T}, (0, 1)),
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
) where {T <: AbstractFloat, AT <: AbstractArray}
    CondPlanar{T, AT}(nn, nvars, basedist, tspan, differentiation_backend)
end

function augmented_f(
    icnf::CondPlanar{T, AT},
    mode::Mode,
    ys::AbstractVector{<:Real},
    st::Any;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1

    function f_aug(u, p, t)
        z = u[1:(end - 1)]
        mz, _ = LuxCore.apply(icnf.nn, vcat(z, ys), p, st)
        trace_J =
            p.u ⋅ transpose(
                only(
                    AbstractDifferentiation.jacobian(
                        differentiation_backend,
                        x -> first(pl_h(icnf.nn, vcat(x, ys), p, st)),
                        z,
                    ),
                ),
            )
        vcat(mz, -trace_J)
    end
    f_aug
end

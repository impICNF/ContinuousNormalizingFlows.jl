export RNODE

"""
Implementation of RNODE from

[Finlay, Chris, Jörn-Henrik Jacobsen, Levon Nurbekyan, and Adam M. Oberman. "How to train your neural ODE: the world of Jacobian and kinetic regularization." arXiv preprint arXiv:2002.02798 (2020).](https://arxiv.org/abs/2002.02798)
"""
struct RNODE{T <: AbstractFloat, AT <: AbstractArray, CM <: ComputeMode} <:
       AbstractICNF{T, AT, CM}
    nn::LuxCore.AbstractExplicitLayer

    nvars::Integer
    basedist::Distribution
    tspan::Tuple{T, T}

    differentiation_backend::AbstractDifferentiation.AbstractBackend

    sol_args::Tuple
    sol_kwargs::Dict

    # trace_test
    # trace_train
end

function augmented_f(
    icnf::RNODE{T, AT, <:ADVectorMode},
    mode::TestMode,
    st::Any;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1

    function f_aug(u, p, t)
        z = u[1:(end - n_aug)]
        ż, J = AbstractDifferentiation.value_and_jacobian(
            differentiation_backend,
            x -> first(LuxCore.apply(icnf.nn, x, p, st)),
            z,
        )
        l̇ = tr(only(J))
        vcat(ż, -l̇)
    end
    f_aug
end

function augmented_f(
    icnf::RNODE{T, AT, CM},
    mode::TestMode,
    st::Any,
    n_batch::Integer;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray, CM <: MatrixMode}
    n_aug = n_augment(icnf, mode) + 1

    function f_aug(u, p, t)
        z = u[1:(end - n_aug), :]
        ż, J = jacobian_batched(x -> first(LuxCore.apply(icnf.nn, x, p, st)), z, T, AT, CM)
        l̇ = transpose(tr.(eachslice(J; dims = 3)))
        vcat(ż, -l̇)
    end
    f_aug
end

function augmented_f(
    icnf::RNODE{T, AT, <:ADVectorMode},
    mode::TrainMode,
    st::Any;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1
    ϵ = convert(AT, randn(rng, T, icnf.nvars))

    function f_aug(u, p, t)
        z = u[1:(end - n_aug)]
        v_pb = AbstractDifferentiation.value_and_pullback_function(
            differentiation_backend,
            x -> first(LuxCore.apply(icnf.nn, x, p, st)),
            z,
        )
        ż, ϵJ = v_pb(ϵ)
        ϵJ = only(ϵJ)
        l̇ = ϵJ ⋅ ϵ
        Ė = norm(ż)
        ṅ = norm(ϵJ)
        vcat(ż, -l̇, Ė, ṅ)
    end
    f_aug
end

function augmented_f(
    icnf::RNODE{T, AT, <:ZygoteMatrixMode},
    mode::TrainMode,
    st::Any,
    n_batch::Integer;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1
    ϵ = convert(AT, randn(rng, T, icnf.nvars, n_batch))

    function f_aug(u, p, t)
        z = u[1:(end - n_aug), :]
        ż, back = Zygote.pullback(x -> first(LuxCore.apply(icnf.nn, x, p, st)), z)
        ϵJ = only(back(ϵ))
        l̇ = sum(ϵJ .* ϵ; dims = 1)
        Ė = transpose(norm.(eachcol(ż)))
        ṅ = transpose(norm.(eachcol(ϵJ)))
        vcat(ż, -l̇, Ė, ṅ)
    end
    f_aug
end

function augmented_f(
    icnf::RNODE{T, AT, <:SDVecJacMatrixMode},
    mode::TrainMode,
    st::Any,
    n_batch::Integer;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1
    ϵ = convert(AT, randn(rng, T, icnf.nvars, n_batch))

    function f_aug(u, p, t)
        z = u[1:(end - n_aug), :]
        ż = first(LuxCore.apply(icnf.nn, z, p, st))
        ϵJ = reshape(
            auto_vecjac(x -> first(LuxCore.apply(icnf.nn, x, p, st)), z, ϵ),
            size(z),
        )
        l̇ = sum(ϵJ .* ϵ; dims = 1)
        Ė = transpose(norm.(eachcol(ż)))
        ṅ = transpose(norm.(eachcol(ϵJ)))
        vcat(ż, -l̇, Ė, ṅ)
    end
    f_aug
end

function augmented_f(
    icnf::RNODE{T, AT, <:SDJacVecMatrixMode},
    mode::TrainMode,
    st::Any,
    n_batch::Integer;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    n_aug = n_augment(icnf, mode) + 1
    ϵ = convert(AT, randn(rng, T, icnf.nvars, n_batch))

    function f_aug(u, p, t)
        z = u[1:(end - n_aug), :]
        ż = first(LuxCore.apply(icnf.nn, z, p, st))
        Jϵ = reshape(
            auto_jacvec(x -> first(LuxCore.apply(icnf.nn, x, p, st)), z, ϵ),
            size(z),
        )
        l̇ = sum(ϵ .* Jϵ; dims = 1)
        Ė = transpose(norm.(eachcol(ż)))
        ṅ = transpose(norm.(eachcol(Jϵ)))
        vcat(ż, -l̇, Ė, ṅ)
    end
    f_aug
end

function loss(
    icnf::RNODE{T, AT, <:VectorMode},
    xs::AbstractVector{<:Real},
    ps::Any,
    st::Any,
    λ₁::T = convert(T, 1e-2),
    λ₂::T = convert(T, 1e-2);
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    mode::Mode = TrainMode(),
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x, Ė, ṅ = inference(icnf, mode, xs, ps, st; differentiation_backend, rng, sol_args, sol_kwargs)
    -logp̂x + λ₁ * Ė + λ₂ * ṅ
end

function loss(
    icnf::RNODE{T, AT, <:MatrixMode},
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::Any,
    λ₁::T = convert(T, 1e-2),
    λ₂::T = convert(T, 1e-2);
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    mode::Mode = TrainMode(),
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    logp̂x, Ė, ṅ = inference(icnf, mode, xs, ps, st; differentiation_backend, rng, sol_args, sol_kwargs)
    mean(-logp̂x + λ₁ * Ė + λ₂ * ṅ)
end

function n_augment(
    icnf::RNODE{T, AT},
    mode::TrainMode,
)::Integer where {T <: AbstractFloat, AT <: AbstractArray}
    2
end

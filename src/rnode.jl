export RNODE

"""
Implementation of RNODE from

[Finlay, Chris, Jörn-Henrik Jacobsen, Levon Nurbekyan, and Adam M. Oberman. "How to train your neural ODE: the world of Jacobian and kinetic regularization." arXiv preprint arXiv:2002.02798 (2020).](https://arxiv.org/abs/2002.02798)
"""
struct RNODE{
    T <: AbstractFloat,
    AT <: AbstractArray,
    CM <: ComputeMode,
    AUGMENTED,
    STEER,
} <: AbstractICNF{T, AT, CM, AUGMENTED, STEER}
    nn::LuxCore.AbstractExplicitLayer
    nvars::Integer
    naugmented::Integer

    basedist::Distribution
    tspan::NTuple{2, T}
    steer_rate::T
    differentiation_backend::AbstractDifferentiation.AbstractBackend
    sol_args::Tuple
    sol_kwargs::Dict
    λ₁::T
    λ₂::T
end

function construct(
    aicnf::Type{<:RNODE},
    nn,
    nvars::Integer,
    naugmented::Integer = 0;
    data_type::Type{<:AbstractFloat} = Float32,
    array_type::Type{<:AbstractArray} = Array,
    compute_mode::Type{<:ComputeMode} = ADVectorMode,
    augmented::Bool = false,
    steer::Bool = false,
    basedist::Distribution = MvNormal(
        Zeros{data_type}(nvars + naugmented),
        Eye{data_type}(nvars + naugmented),
    ),
    tspan::NTuple{2} = (zero(data_type), one(data_type)),
    steer_rate::AbstractFloat = zero(data_type),
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    sol_args::Tuple = (),
    sol_kwargs::Dict = Dict(
        :alg_hints => [:nonstiff, :memorybound],
        :reltol => 1e-2 + eps(1e-2),
    ),
    λ₁::AbstractFloat = convert(data_type, 1e-2),
    λ₂::AbstractFloat = convert(data_type, 1e-2),
)
    !augmented && !iszero(naugmented) && error("'naugmented' > 0: 'augmented' must be true")
    !steer && !iszero(steer_rate) && error("'steer_rate' > 0: 'steer' must be true")

    aicnf{data_type, array_type, compute_mode, augmented, steer}(
        nn,
        nvars,
        naugmented,
        basedist,
        tspan,
        steer_rate,
        differentiation_backend,
        sol_args,
        sol_kwargs,
        λ₁,
        λ₂,
    )
end

function augmented_f(
    icnf::RNODE{<:AbstractFloat, <:AbstractArray, <:ADVectorMode},
    mode::TrainMode,
    st::Any;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode) + 1
    ϵ = randn_T_AT(icnf, rng, icnf.nvars)

    function f_aug(u, p, t)
        z = @view u[begin:(end - n_aug)]
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
    icnf::RNODE{<:AbstractFloat, <:AbstractArray, <:ZygoteMatrixMode},
    mode::TrainMode,
    st::Any,
    n_batch::Integer;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode) + 1
    ϵ = randn_T_AT(icnf, rng, icnf.nvars, n_batch)

    function f_aug(u, p, t)
        z = @view u[begin:(end - n_aug), :]
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
    icnf::RNODE{<:AbstractFloat, <:AbstractArray, <:SDVecJacMatrixMode},
    mode::TrainMode,
    st::Any,
    n_batch::Integer;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode) + 1
    ϵ = randn_T_AT(icnf, rng, icnf.nvars, n_batch)

    function f_aug(u, p, t)
        z = @view u[begin:(end - n_aug), :]
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
    icnf::RNODE{<:AbstractFloat, <:AbstractArray, <:SDJacVecMatrixMode},
    mode::TrainMode,
    st::Any,
    n_batch::Integer;
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode) + 1
    ϵ = randn_T_AT(icnf, rng, icnf.nvars, n_batch)

    function f_aug(u, p, t)
        z = @view u[begin:(end - n_aug), :]
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

@inline function loss(
    icnf::RNODE{<:AbstractFloat, <:AbstractArray, <:VectorMode},
    mode::TrainMode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::Any;
    tspan::NTuple{2} = icnf.tspan,
    steer_rate::AbstractFloat = icnf.steer_rate,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
    λ₁::AbstractFloat = icnf.λ₁,
    λ₂::AbstractFloat = icnf.λ₂,
)
    logp̂x, Ė, ṅ = inference(
        icnf,
        mode,
        xs,
        ps,
        st;
        tspan,
        steer_rate,
        basedist,
        differentiation_backend,
        rng,
        sol_args,
        sol_kwargs,
    )
    -logp̂x + λ₁ * Ė + λ₂ * ṅ
end

@inline function loss(
    icnf::RNODE{<:AbstractFloat, <:AbstractArray, <:MatrixMode},
    mode::TrainMode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::Any;
    tspan::NTuple{2} = icnf.tspan,
    steer_rate::AbstractFloat = icnf.steer_rate,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
    λ₁::AbstractFloat = icnf.λ₁,
    λ₂::AbstractFloat = icnf.λ₂,
)
    logp̂x, Ė, ṅ = inference(
        icnf,
        mode,
        xs,
        ps,
        st;
        tspan,
        steer_rate,
        basedist,
        differentiation_backend,
        rng,
        sol_args,
        sol_kwargs,
    )
    mean(-logp̂x + λ₁ * Ė + λ₂ * ṅ)
end

@inline function n_augment(::RNODE, ::TrainMode)
    2
end

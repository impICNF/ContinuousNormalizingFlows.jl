export RNODE

"""
Implementation of RNODE from

[Finlay, Chris, Jörn-Henrik Jacobsen, Levon Nurbekyan, and Adam M. Oberman. "How to train your neural ODE: the world of Jacobian and kinetic regularization." arXiv preprint arXiv:2002.02798 (2020).](https://arxiv.org/abs/2002.02798)
"""
struct RNODE{T <: AbstractFloat, CM <: ComputeMode, AUGMENTED, STEER} <:
       AbstractICNF{T, CM, AUGMENTED, STEER}
    nn::LuxCore.AbstractExplicitLayer
    nvars::Integer
    naugmented::Integer

    resource::AbstractResource
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
    compute_mode::Type{<:ComputeMode} = ADVectorMode,
    augmented::Bool = false,
    steer::Bool = false,
    resource::AbstractResource = CPU1(),
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

    aicnf{data_type, compute_mode, augmented, steer}(
        nn,
        nvars,
        naugmented,
        resource,
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
    u,
    p,
    t,
    icnf::RNODE{<:AbstractFloat, <:ADVectorMode},
    mode::TrainMode,
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    ϵ = randn_T_AT(resource, icnf, rng, icnf.nvars + n_aug_input)
    z = @view u[begin:(end - n_aug - 1)]
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

function augmented_f(
    u,
    p,
    t,
    icnf::RNODE{<:AbstractFloat, <:ZygoteMatrixMode},
    mode::TrainMode,
    st::Any,
    n_batch::Integer;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    ϵ = randn_T_AT(resource, icnf, rng, icnf.nvars + n_aug_input, n_batch)
    z = @view u[begin:(end - n_aug - 1), :]
    ż, back = Zygote.pullback(x -> first(LuxCore.apply(icnf.nn, x, p, st)), z)
    ϵJ = only(back(ϵ))
    l̇ = sum(ϵJ .* ϵ; dims = 1)
    Ė = transpose(norm.(eachcol(ż)))
    ṅ = transpose(norm.(eachcol(ϵJ)))
    vcat(ż, -l̇, Ė, ṅ)
end

function augmented_f(
    u,
    p,
    t,
    icnf::RNODE{<:AbstractFloat, <:SDVecJacMatrixMode},
    mode::TrainMode,
    st::Any,
    n_batch::Integer;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    ϵ = randn_T_AT(resource, icnf, rng, icnf.nvars + n_aug_input, n_batch)
    z = @view u[begin:(end - n_aug - 1), :]
    ż = first(LuxCore.apply(icnf.nn, z, p, st))
    ϵJ = reshape(auto_vecjac(x -> first(LuxCore.apply(icnf.nn, x, p, st)), z, ϵ), size(z))
    l̇ = sum(ϵJ .* ϵ; dims = 1)
    Ė = transpose(norm.(eachcol(ż)))
    ṅ = transpose(norm.(eachcol(ϵJ)))
    vcat(ż, -l̇, Ė, ṅ)
end

function augmented_f(
    u,
    p,
    t,
    icnf::RNODE{<:AbstractFloat, <:SDJacVecMatrixMode},
    mode::TrainMode,
    st::Any,
    n_batch::Integer;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    ϵ = randn_T_AT(resource, icnf, rng, icnf.nvars + n_aug_input, n_batch)
    z = @view u[begin:(end - n_aug - 1), :]
    ż = first(LuxCore.apply(icnf.nn, z, p, st))
    Jϵ = reshape(auto_jacvec(x -> first(LuxCore.apply(icnf.nn, x, p, st)), z, ϵ), size(z))
    l̇ = sum(ϵ .* Jϵ; dims = 1)
    Ė = transpose(norm.(eachcol(ż)))
    ṅ = transpose(norm.(eachcol(Jϵ)))
    vcat(ż, -l̇, Ė, ṅ)
end

@inline function loss(
    icnf::RNODE{<:AbstractFloat, <:VectorMode},
    mode::TrainMode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steer_rate::AbstractFloat = steer_rate_value(icnf),
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
    icnf::RNODE{<:AbstractFloat, <:MatrixMode},
    mode::TrainMode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steer_rate::AbstractFloat = steer_rate_value(icnf),
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

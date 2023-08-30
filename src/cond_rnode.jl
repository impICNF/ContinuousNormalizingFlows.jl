export CondRNODE

"""
Implementation of RNODE (Conditional Version)
"""
struct CondRNODE{
    T <: AbstractFloat,
    CM <: ComputeMode,
    AUGMENTED,
    STEER,
    NN <: LuxCore.AbstractExplicitLayer,
    RESOURCE <: AbstractResource,
    BASEDIST <: Distribution,
    TSPAN <: NTuple{2, T},
    STEERDIST <: Distribution,
    DIFFERENTIATION_BACKEND <: AbstractDifferentiation.AbstractBackend,
    AUTODIFF_BACKEND <: ADTypes.AbstractADType,
    _FNN <: Function,
} <: AbstractCondICNF{T, CM, AUGMENTED, STEER}
    nn::NN
    nvars::Int
    naugmented::Int

    resource::RESOURCE
    basedist::BASEDIST
    tspan::TSPAN
    steerdist::STEERDIST
    differentiation_backend::DIFFERENTIATION_BACKEND
    autodiff_backend::AUTODIFF_BACKEND
    sol_args::Tuple
    sol_kwargs::Dict
    _fnn::_FNN
    λ₁::T
    λ₂::T
end

function construct(
    aicnf::Type{<:CondRNODE},
    nn,
    nvars::Int,
    naugmented::Int = 0;
    data_type::Type{<:AbstractFloat} = Float32,
    compute_mode::Type{<:ComputeMode} = ADVectorMode,
    resource::AbstractResource = CPU1(),
    basedist::Distribution = MvNormal(
        Zeros{data_type}(nvars + naugmented),
        Eye{data_type}(nvars + naugmented),
    ),
    tspan::NTuple{2} = (zero(data_type), one(data_type)),
    steer_rate::AbstractFloat = zero(data_type),
    differentiation_backend::AbstractDifferentiation.AbstractBackend = AbstractDifferentiation.ZygoteBackend(),
    autodiff_backend::ADTypes.AbstractADType = AutoZygote(),
    sol_args::Tuple = (),
    sol_kwargs::Dict = Dict(
        :alg_hints => [:nonstiff, :memorybound],
        :reltol => 1e-2 + eps(1e-2),
    ),
    λ₁::AbstractFloat = convert(data_type, 1e-2),
    λ₂::AbstractFloat = convert(data_type, 1e-2),
)
    steerdist = Uniform{data_type}(-steer_rate, steer_rate)
    _fnn(x, ps, st) = first(nn(x, ps, st))

    aicnf{
        data_type,
        compute_mode,
        !iszero(naugmented),
        !iszero(steer_rate),
        typeof(nn),
        typeof(resource),
        typeof(basedist),
        typeof(tspan),
        typeof(steerdist),
        typeof(differentiation_backend),
        typeof(autodiff_backend),
        typeof(_fnn),
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
        sol_args,
        sol_kwargs,
        _fnn,
        λ₁,
        λ₂,
    )
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondRNODE{<:AbstractFloat, <:ADVectorMode},
    mode::TrainMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1)]
    v_pb = AbstractDifferentiation.value_and_pullback_function(
        differentiation_backend,
        x -> icnf._fnn(cat(x, ys; dims = 1), p, st),
        z,
    )
    ż, ϵJ = v_pb(ϵ)
    ϵJ = only(ϵJ)
    l̇ = ϵJ ⋅ ϵ
    Ė = norm(ż)
    ṅ = norm(ϵJ)
    cat(ż, -l̇, Ė, ṅ; dims = 1)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondRNODE{<:AbstractFloat, <:ZygoteMatrixMode},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż, back = Zygote.pullback(icnf._fnn, cat(z, ys; dims = 1), p, st)
    ϵJ = first(back(ϵ))
    l̇ = sum(ϵJ .* ϵ; dims = 1)
    Ė = transpose(norm.(eachcol(ż)))
    ṅ = transpose(norm.(eachcol(ϵJ)))
    cat(ż, -l̇, Ė, ṅ; dims = 1)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondRNODE{<:AbstractFloat, <:SDVecJacMatrixMode},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż = icnf._fnn(cat(z, ys; dims = 1), p, st)
    Jf = VecJac(x -> icnf._fnn(cat(x, ys; dims = 1), p, st), z; autodiff = autodiff_backend)
    ϵJ = reshape(Jf * ϵ, size(z))
    l̇ = sum(ϵJ .* ϵ; dims = 1)
    Ė = transpose(norm.(eachcol(ż)))
    ṅ = transpose(norm.(eachcol(ϵJ)))
    cat(ż, -l̇, Ė, ṅ; dims = 1)
end

@views function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::CondRNODE{<:AbstractFloat, <:SDJacVecMatrixMode},
    mode::TrainMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = u[begin:(end - n_aug - 1), :]
    ż = icnf._fnn(cat(z, ys; dims = 1), p, st)
    Jf = JacVec(x -> icnf._fnn(cat(x, ys; dims = 1), p, st), z; autodiff = autodiff_backend)
    Jϵ = reshape(Jf * ϵ, size(z))
    l̇ = sum(ϵ .* Jϵ; dims = 1)
    Ė = transpose(norm.(eachcol(ż)))
    ṅ = transpose(norm.(eachcol(Jϵ)))
    cat(ż, -l̇, Ė, ṅ; dims = 1)
end

@inline function loss(
    icnf::CondRNODE{<:AbstractFloat, <:VectorMode},
    mode::TrainMode,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
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
        ys,
        ps,
        st;
        resource,
        tspan,
        steerdist,
        basedist,
        differentiation_backend,
        autodiff_backend,
        rng,
        sol_args,
        sol_kwargs,
    )
    -logp̂x + λ₁ * Ė + λ₂ * ṅ
end

@inline function loss(
    icnf::CondRNODE{<:AbstractFloat, <:MatrixMode},
    mode::TrainMode,
    xs::AbstractMatrix{<:Real},
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    autodiff_backend::ADTypes.AbstractADType = icnf.autodiff_backend,
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
        ys,
        ps,
        st;
        resource,
        tspan,
        steerdist,
        basedist,
        differentiation_backend,
        autodiff_backend,
        rng,
        sol_args,
        sol_kwargs,
    )
    mean(-logp̂x + λ₁ * Ė + λ₂ * ṅ)
end

@inline function n_augment(::CondRNODE, ::TrainMode)
    2
end

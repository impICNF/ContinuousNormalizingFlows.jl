export
    inference, generate, TestMode, TrainMode,
    loss_f, cb_f,
    ICNFModel, ICNFDistribution

abstract type Flows end
abstract type NormalizingFlows <: Flows end
abstract type ContinuousNormalizingFlows <: NormalizingFlows end
abstract type InfinitesimalContinuousNormalizingFlows <: ContinuousNormalizingFlows end
abstract type AbstractICNF <: InfinitesimalContinuousNormalizingFlows where {T <: AbstractFloat} end

abstract type Mode end
struct TestMode <: Mode end
struct TrainMode <: Mode end

function inference(icnf::AbstractICNF, mode::TestMode, xs::AbstractMatrix{T})::AbstractVector{T} where {T <: AbstractFloat} end
function inference(icnf::AbstractICNF, mode::TrainMode, xs::AbstractMatrix{T})::AbstractVector{T} where {T <: AbstractFloat} end

function generate(icnf::AbstractICNF, mode::TestMode, n::Integer; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat} end
function generate(icnf::AbstractICNF, mode::TrainMode, n::Integer; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat} end

default_acceleration = CPU1()
default_solver_test = Feagin14()
default_solver_train = Tsit5(; thread=OrdinaryDiffEq.True())
default_sensealg = InterpolatingAdjoint(
    ;
    autodiff=true,
    chunk_size=0,
    autojacvec=ZygoteVJP(),
)

# Flux interface

function loss_f(icnf::AbstractICNF; agg::Function=mean)::Function where {T <: AbstractFloat} end

function (m::AbstractICNF)(x::AbstractMatrix{T})::AbstractVector{T} where {T <: AbstractFloat}
    inference(m, TestMode(), x)
end

function cb_f(loss::Function, data::AbstractVector{T2})::Function where {T <: AbstractFloat, T2 <: AbstractMatrix{T}}
    xs = first(data)
    function f()::Nothing
        @info "loss = $(loss(xs))"
    end
    f
end

# MLJ interface

abstract type MLJICNF <: MLJModelInterface.Unsupervised end

@with_kw mutable struct ICNFModel{T2} <: MLJICNF where {T <: AbstractFloat, T2 <: AbstractICNF}
    m::T2
    loss::Function = loss_f(m)

    optimizer::Flux.Optimise.AbstractOptimiser = AMSGrad()
    n_epochs::Integer = 128

    batch_size::Integer = 128

    cb_timeout::Integer = 16
end

function ICNFModel(
        nvars::Integer,
        n_hidden_ratio::Integer=4,
        ;
        optimizer::Flux.Optimise.AbstractOptimiser=AMSGrad(),
        n_epochs::Integer=128,

        batch_size::Integer=128,

        cb_timeout::Integer=16,
        ) where {T <: AbstractFloat, T2 <: AbstractICNF}
    nn = Chain(
        Dense(nvars, nvars*n_hidden_ratio, tanh),
        Dense(nvars*n_hidden_ratio, nvars, tanh),
    )
    m = FFJORD{Float64}(nn, nvars)
    loss = loss_f(m)
    ICNFModel(; m, loss, optimizer, n_epochs, batch_size, cb_timeout)
end

function MLJModelInterface.fit(model::ICNFModel, verbosity, X)
    x = collect(MLJModelInterface.matrix(X)')
    data = broadcast(nx -> hcat(nx...), Base.Iterators.partition(eachcol(x), model.batch_size))

    initial_loss_value = model.loss(x)
    t₀ = time()
    Flux.Optimise.@epochs model.n_epochs Flux.Optimise.train!(model.loss, Flux.params(model.m), data, model.optimizer; cb=Flux.throttle(cb_f(model.loss, data), model.cb_timeout))
    t₁ = time()
    final_loss_value = model.loss(x)
    Δt = t₁ - t₀
    @info "time cost (fit) = $(Δt) seconds"

    fitresult = nothing
    cache = nothing
    report = (
        fit_time=Δt,
        initial_loss_value=initial_loss_value,
        final_loss_value=final_loss_value,
    )
    fitresult, cache, report
end

function MLJModelInterface.transform(model::ICNFModel, fitresult, Xnew)
    xnew = collect(MLJModelInterface.matrix(Xnew)')

    t₀ = time()
    logp̂x = inference(model.m, TestMode(), xnew)
    t₁ = time()
    Δt = t₁ - t₀
    @info "time cost (transform) = $(Δt) seconds"

    DataFrame(px=exp.(logp̂x))
end

function MLJModelInterface.fitted_params(model::ICNFModel, fitresult)
    (
        learned_parameters=model.m.p,
    )
end

MLJBase.metadata_pkg(
    ICNFModel,
    package_name="ICNF",
    package_uuid="9bd0f7d2-bd29-441d-bcde-0d11364d2762",
    package_url="https://github.com/impICNF/ICNF.jl",
    is_pure_julia=true,
    package_license="MIT",
    is_wrapper=false,
)

MLJBase.metadata_model(
    ICNFModel,
    input_scitype=Table{AbstractVector{ScientificTypes.Continuous}},
    target_scitype=Table{AbstractVector{ScientificTypes.Continuous}},
    output_scitype=Table{AbstractVector{ScientificTypes.Continuous}},
    supports_weights=false,
    docstring="ICNFModel",
    load_path="ICNF.ICNFModel",
)

# Distributions interface

@with_kw struct ICNFDistribution{T2} <: ContinuousMultivariateDistribution where {T <: AbstractFloat, T2 <: AbstractICNF}
    m::T2
end

Base.length(d::ICNFDistribution) = d.m.nvars
Base.eltype(d::ICNFDistribution) = eltype(d.m.p)
Distributions._logpdf(d::ICNFDistribution, x::AbstractVector) = first(Distributions._logpdf(d, hcat(x)))
Distributions._logpdf(d::ICNFDistribution, A::AbstractMatrix) = inference(d.m, TestMode(), A)
Distributions._rand!(rng::AbstractRNG, d::ICNFDistribution, x::AbstractVector) = (x[:] = generate(d.m, TestMode(), size(x, 2); rng))
Distributions._rand!(rng::AbstractRNG, d::ICNFDistribution, A::AbstractMatrix) = (A[:] = generate(d.m, TestMode(), size(A, 2); rng))

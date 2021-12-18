export
    inference, generate, TestMode, TrainMode,
    loss_f, cb_f,
    ICNFModel, ICNFDistribution

abstract type Flows end
abstract type NormalizingFlows <: Flows end
abstract type ContinuousNormalizingFlows <: NormalizingFlows end
abstract type InfinitesimalContinuousNormalizingFlows <: ContinuousNormalizingFlows end

abstract type Mode end
struct TestMode <: Mode end
struct TrainMode <: Mode end

default_acceleration = CPU1()
default_solver_test = Feagin14()
default_solver_train = Tsit5(; thread=OrdinaryDiffEq.True())
default_sensealg = InterpolatingAdjoint(
    ;
    autodiff=true,
    chunk_size=0,
    autojacvec=ZygoteVJP(),
)

# - orginal config

abstract type AbstractICNF{T} <: InfinitesimalContinuousNormalizingFlows where {T <: AbstractFloat} end

function inference(icnf::AbstractICNF{T}, mode::TestMode, xs::AbstractMatrix{T})::AbstractVector{T} where {T <: AbstractFloat} end
function inference(icnf::AbstractICNF{T}, mode::TrainMode, xs::AbstractMatrix{T})::AbstractVector{T} where {T <: AbstractFloat} end

function generate(icnf::AbstractICNF{T}, mode::TestMode, n::Integer; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat} end
function generate(icnf::AbstractICNF{T}, mode::TrainMode, n::Integer; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat} end

function loss_f(icnf::AbstractICNF{T}; agg::Function=mean)::Function where {T <: AbstractFloat} end

# -- Flux interface

function (m::AbstractICNF{T})(x::AbstractMatrix{T})::AbstractVector{T} where {T <: AbstractFloat}
    inference(m, TestMode(), x)
end

function cb_f(icnf::AbstractICNF{T}, loss::Function, data::AbstractVector{T2})::Function where {T <: AbstractFloat, T2 <: AbstractMatrix{T}}
    xs = first(data)
    function f()::Nothing
        @info "loss = $(loss(xs))"
    end
    f
end

# - conditional config

abstract type AbstractCondICNF{T} <: InfinitesimalContinuousNormalizingFlows where {T <: AbstractFloat} end

function inference(icnf::AbstractCondICNF{T}, mode::TestMode, xs::AbstractMatrix{T}, ys::AbstractMatrix{T})::AbstractVector{T} where {T <: AbstractFloat} end
function inference(icnf::AbstractCondICNF{T}, mode::TrainMode, xs::AbstractMatrix{T}, ys::AbstractMatrix{T})::AbstractVector{T} where {T <: AbstractFloat} end

function generate(icnf::AbstractCondICNF{T}, mode::TestMode, ys::AbstractMatrix{T}, n::Integer; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat} end
function generate(icnf::AbstractCondICNF{T}, mode::TrainMode, ys::AbstractMatrix{T}, n::Integer; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat} end

function loss_f(icnf::AbstractCondICNF{T}; agg::Function=mean)::Function where {T <: AbstractFloat} end

# -- Flux interface

function (m::AbstractCondICNF{T})(x::AbstractMatrix{T}, y::AbstractMatrix{T})::AbstractVector{T} where {T <: AbstractFloat}
    inference(m, TestMode(), x, y)
end

function cb_f(icnf::AbstractCondICNF{T}, loss::Function, data::AbstractVector{T3})::Function where {T <: AbstractFloat, T2 <: AbstractMatrix{T}, T3 <: Tuple{T2, T2}}
    xs, ys = first(data)
    function f()::Nothing
        @info "loss = $(loss(xs, ys))"
    end
    f
end

# MLJ interface

abstract type MLJICNF <: MLJModelInterface.Unsupervised end

mutable struct ICNFModel{T2} <: MLJICNF where {T <: AbstractFloat, T2 <: AbstractICNF{T}}
    m::T2
    loss::Function

    optimizer::Flux.Optimise.AbstractOptimiser
    n_epochs::Integer

    batch_size::Integer

    cb_timeout::Integer
end

function ICNFModel(
        m::T2,
        loss::Function = loss_f(m),
        ;
        optimizer::Flux.Optimise.AbstractOptimiser=AMSGrad(),
        n_epochs::Integer=128,

        batch_size::Integer=128,

        cb_timeout::Integer=16,
        ) where {T <: AbstractFloat, T2 <: AbstractICNF{T}}
    ICNFModel(m, loss, optimizer, n_epochs, batch_size, cb_timeout)
end

function MLJModelInterface.fit(model::ICNFModel, verbosity, X)
    x = collect(MLJModelInterface.matrix(X)')
    data = broadcast(nx -> hcat(nx...), Base.Iterators.partition(eachcol(x), model.batch_size))

    initial_loss_value = model.loss(x)
    t₀ = time()
    Flux.Optimise.@epochs model.n_epochs Flux.Optimise.train!(model.loss, Flux.params(model.m), data, model.optimizer; cb=Flux.throttle(cb_f(model.m, model.loss, data), model.cb_timeout))
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

struct ICNFDistribution{T2} <: ContinuousMultivariateDistribution where {T <: AbstractFloat, T2 <: AbstractICNF{T}}
    m::T2
end

Base.length(d::ICNFDistribution) = d.m.nvars
Base.eltype(d::ICNFDistribution) = eltype(d.m.p)
Distributions._logpdf(d::ICNFDistribution, x::AbstractVector) = first(Distributions._logpdf(d, hcat(x)))
Distributions._logpdf(d::ICNFDistribution, A::AbstractMatrix) = inference(d.m, TestMode(), A)
Distributions._rand!(rng::AbstractRNG, d::ICNFDistribution, x::AbstractVector) = (x[:] = generate(d.m, TestMode(), size(x, 2); rng))
Distributions._rand!(rng::AbstractRNG, d::ICNFDistribution, A::AbstractMatrix) = (A[:] = generate(d.m, TestMode(), size(A, 2); rng))

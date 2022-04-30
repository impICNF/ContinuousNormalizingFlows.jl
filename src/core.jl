export
    inference, generate, loss,
    TestMode, TrainMode,
    FluxOptApp, OptimOptApp, SciMLOptApp,
    loss_f, cb_f,
    ICNFModel, CondICNFModel, ICNFDistribution

abstract type Flows end
abstract type NormalizingFlows <: Flows end
abstract type ContinuousNormalizingFlows <: NormalizingFlows end
abstract type InfinitesimalContinuousNormalizingFlows <: ContinuousNormalizingFlows end

abstract type Mode end
struct TestMode <: Mode end
struct TrainMode <: Mode end

abstract type OptApp end
struct FluxOptApp <: OptApp end
struct OptimOptApp <: OptApp end
struct SciMLOptApp <: OptApp end

default_acceleration = CPU1()
default_solver_test = Feagin14()
default_solver_train = Tsit5(; thread=OrdinaryDiffEq.True())
default_sensealg = InterpolatingAdjoint(
    ;
    autodiff=true,
    chunk_size=0,
    autojacvec=ZygoteVJP(),
)
default_optimizer = Dict(
    FluxOptApp => Flux.AMSGrad(),
    OptimOptApp => ConjugateGradient(),
    SciMLOptApp => Flux.AMSGrad(),
)

# - orginal config

abstract type AbstractICNF{T} <: InfinitesimalContinuousNormalizingFlows where {T <: AbstractFloat} end

function inference(icnf::AbstractICNF{T}, mode::TestMode, xs::AbstractMatrix{T}, p::AbstractVector=icnf.p)::AbstractVector where {T <: AbstractFloat} end
function inference(icnf::AbstractICNF{T}, mode::TrainMode, xs::AbstractMatrix{T}, p::AbstractVector=icnf.p)::AbstractVector where {T <: AbstractFloat} end

function generate(icnf::AbstractICNF{T}, mode::TestMode, n::Integer, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat} end
function generate(icnf::AbstractICNF{T}, mode::TrainMode, n::Integer, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat} end

function loss(icnf::AbstractICNF{T}, xs::AbstractMatrix{T}, p::AbstractVector=icnf.p; agg::Function=mean) where {T <: AbstractFloat} end

# -- Flux interface

function (icnf::AbstractICNF{T})(xs::AbstractMatrix{T})::AbstractVector{T} where {T <: AbstractFloat}
    inference(icnf, TestMode(), xs)
end

function loss_f(icnf::AbstractICNF{T}, opt_app::FluxOptApp)::Function where {T <: AbstractFloat}
    function f(xs::AbstractMatrix{T})::T
        loss(icnf, xs)
    end
    f
end

function cb_f(icnf::AbstractICNF{T}, opt_app::FluxOptApp, loss::Function, data::DataLoader{T3})::Function where {T <: AbstractFloat, T2 <: AbstractMatrix{T}, T3 <: Tuple{T2}}
    xs, = first(data)
    function f()::Nothing
        vl = loss(icnf, xs)
        @info "loss = $vl"
        nothing
    end
    f
end

# -- SciML interface

function loss_f(icnf::AbstractICNF{T}, opt_app::SciMLOptApp)::Function where {T <: AbstractFloat}
    function f(p::AbstractVector, θ::SciMLBase.NullParameters, xs::AbstractMatrix{T})
        loss(icnf, xs, p)
    end
    f
end

function cb_f(icnf::AbstractICNF{T}, opt_app::SciMLOptApp, loss::Function, data::DataLoader{T3})::Function where {T <: AbstractFloat, T2 <: AbstractMatrix{T}, T3 <: Tuple{T2}}
    xs, = first(data)
    function f(p::AbstractVector{T}, l::T)::Bool
        vl = loss(icnf, xs, p)
        @info "loss = $vl"
        false
    end
    f
end

# -- Optim interface

function loss_f(icnf::AbstractICNF{T}, opt_app::OptimOptApp, itrtr::AbstractVector)::Function where {T <: AbstractFloat}
    function f(p::AbstractVector{T})::T
        xs, = itrtr[1]
        loss(icnf, xs, p)
    end
    f
end

function cb_f(icnf::AbstractICNF{T}, opt_app::OptimOptApp, loss::Function, data::DataLoader{T3}, itrtr::AbstractVector)::Function where {T <: AbstractFloat, T2 <: AbstractMatrix{T}, T3 <: Tuple{T2}}
    xs, = first(data)
    function f(s::OptimizationState)::Bool
        vl = loss(icnf, xs, s.metadata["x"])
        @info "loss = $vl"
        nxitr = iterate(data, itrtr[2])
        if isnothing(nxitr)
            true
        else
            itrtr .= nxitr
            false
        end
    end
    f
end

# - conditional config

abstract type AbstractCondICNF{T} <: InfinitesimalContinuousNormalizingFlows where {T <: AbstractFloat} end

function inference(icnf::AbstractCondICNF{T}, mode::TestMode, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p)::AbstractVector where {T <: AbstractFloat} end
function inference(icnf::AbstractCondICNF{T}, mode::TrainMode, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p)::AbstractVector where {T <: AbstractFloat} end

function generate(icnf::AbstractCondICNF{T}, mode::TestMode, ys::AbstractMatrix{T}, n::Integer, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat} end
function generate(icnf::AbstractCondICNF{T}, mode::TrainMode, ys::AbstractMatrix{T}, n::Integer, p::AbstractVector=icnf.p; rng::Union{AbstractRNG, Nothing}=nothing)::AbstractMatrix{T} where {T <: AbstractFloat} end

function loss(icnf::AbstractICNF{T}, xs::AbstractMatrix{T}, ys::AbstractMatrix{T}, p::AbstractVector=icnf.p; agg::Function=mean) where {T <: AbstractFloat} end

# -- Flux interface

function (icnf::AbstractCondICNF{T})(xs::AbstractMatrix{T}, ys::AbstractMatrix{T})::AbstractVector{T} where {T <: AbstractFloat}
    inference(icnf, TestMode(), xs, ys)
end

function loss_f(icnf::AbstractCondICNF{T}, opt_app::FluxOptApp)::Function where {T <: AbstractFloat}
    function f(xs::AbstractMatrix{T}, ys::AbstractMatrix{T})::T
        loss(icnf, xs, ys)
    end
    f
end

function cb_f(icnf::AbstractCondICNF{T}, opt_app::FluxOptApp, loss::Function, data::DataLoader{T3})::Function where {T <: AbstractFloat, T2 <: AbstractMatrix{T}, T3 <: Tuple{T2, T2}}
    xs, ys = first(data)
    function f()::Nothing
        vl = loss(icnf, xs, ys)
        @info "loss = $vl"
        nothing
    end
    f
end

# -- SciML interface

function loss_f(icnf::AbstractCondICNF{T}, opt_app::SciMLOptApp)::Function where {T <: AbstractFloat}
    function f(p::AbstractVector, θ::SciMLBase.NullParameters, xs::AbstractMatrix{T}, ys::AbstractMatrix{T})
        loss(icnf, xs, ys, p)
    end
    f
end

function cb_f(icnf::AbstractCondICNF{T}, opt_app::SciMLOptApp, loss::Function, data::DataLoader{T3})::Function where {T <: AbstractFloat, T2 <: AbstractMatrix{T}, T3 <: Tuple{T2, T2}}
    xs, ys = first(data)
    function f(p::AbstractVector{T}, l::T)::Bool
        vl = loss(icnf, xs, ys, p)
        @info "loss = $vl"
        false
    end
    f
end

# -- Optim interface

function loss_f(icnf::AbstractCondICNF{T}, opt_app::OptimOptApp, itrtr::AbstractVector)::Function where {T <: AbstractFloat}
    function f(p::AbstractVector{T})::T
        xs, ys = itrtr[1]
        loss(icnf, xs, ys, p)
    end
    f
end

function cb_f(icnf::AbstractCondICNF{T}, opt_app::OptimOptApp, loss::Function, data::DataLoader{T3}, itrtr::AbstractVector)::Function where {T <: AbstractFloat, T2 <: AbstractMatrix{T}, T3 <: Tuple{T2, T2}}
    xs, ys = first(data)
    function f(s::OptimizationState)::Bool
        vl = loss(icnf, xs, ys, s.metadata["x"])
        @info "loss = $vl"
        nxitr = iterate(data, itrtr[2])
        if isnothing(nxitr)
            true
        else
            itrtr .= nxitr
            false
        end
    end
    f
end

# MLJ interface

abstract type MLJICNF <: MLJModelInterface.Unsupervised end

mutable struct ICNFModel{T2} <: MLJICNF where {T <: AbstractFloat, T2 <: AbstractICNF{T}}
    m::T2
    loss::Function

    opt_app::OptApp
    optimizer::Any
    n_epochs::Integer
    adtype::SciMLBase.AbstractADType

    batch_size::Integer
end

function ICNFModel(
        m::T2,
        loss::Function=loss,
        ;
        opt_app::OptApp=FluxOptApp(),
        optimizer::Any=default_optimizer[typeof(opt_app)],
        n_epochs::Integer=128,
        adtype::SciMLBase.AbstractADType=GalacticOptim.AutoZygote(),

        batch_size::Integer=128,
        ) where {T <: AbstractFloat, T2 <: AbstractICNF{T}}
    ICNFModel(m, loss, opt_app, optimizer, n_epochs, adtype, batch_size)
end

function MLJModelInterface.fit(model::ICNFModel, verbosity, X)
    x = collect(MLJModelInterface.matrix(X)')
    data = DataLoader((x,); batchsize=model.batch_size, shuffle=true, partial=true)
    ncdata = ncycle(data, model.n_epochs)
    initial_loss_value = model.loss(model.m, x)

    if model.opt_app isa FluxOptApp
        @assert model.optimizer isa Flux.Optimise.AbstractOptimiser
        _loss = loss_f(model.m, model.opt_app)
        _cb = cb_f(model.m, model.opt_app, model.loss, data)
        _p = Flux.params(model.m)
        t₀ = time()
        Flux.Optimise.train!(_loss, _p, ncdata, model.optimizer; cb=_cb)
        t₁ = time()
    elseif model.opt_app isa OptimOptApp
        @assert model.optimizer isa Optim.AbstractOptimizer
        itrtr = Any[nothing, nothing]
        itrtr .= iterate(ncdata)
        _loss = loss_f(model.m, model.opt_app, itrtr)
        _cb = cb_f(model.m, model.opt_app, model.loss, data, itrtr)
        ops = Optim.Options(
            x_abstol=-Inf, x_reltol=-Inf,
            f_abstol=-Inf, f_reltol=-Inf,
            g_abstol=-Inf, g_reltol=-Inf,
            outer_x_abstol=-Inf, outer_x_reltol=-Inf,
            outer_f_abstol=-Inf, outer_f_reltol=-Inf,
            outer_g_abstol=-Inf, outer_g_reltol=-Inf,
            f_calls_limit=0, g_calls_limit=0, h_calls_limit=0,
            allow_f_increases=true, allow_outer_f_increases=true,
            successive_f_tol=typemax(Int), iterations=typemax(Int), outer_iterations=typemax(Int),
            store_trace=false, trace_simplex=true, show_trace=false, extended_trace=true,
            show_every=1, callback=_cb, time_limit=Inf,
        )
        t₀ = time()
        res = optimize(_loss, model.m.p, model.optimizer, ops)
        t₁ = time()
        model.m.p .= res.minimizer
    elseif model.opt_app isa SciMLOptApp
        _loss = loss_f(model.m, model.opt_app)
        _cb = cb_f(model.m, model.opt_app, model.loss, data)
        optfunc = OptimizationFunction(_loss, model.adtype)
        optprob = OptimizationProblem(optfunc, model.m.p)
        t₀ = time()
        res = solve(optprob, model.optimizer, ncdata; cb=_cb)
        t₁ = time()
        model.m.p .= res.u
    end
    final_loss_value = model.loss(model.m, x)
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

mutable struct CondICNFModel{T2} <: MLJICNF where {T <: AbstractFloat, T2 <: AbstractCondICNF{T}}
    m::T2
    loss::Function

    opt_app::OptApp
    optimizer::Any
    n_epochs::Integer
    adtype::SciMLBase.AbstractADType

    batch_size::Integer
end

function CondICNFModel(
        m::T2,
        loss::Function=loss,
        ;
        opt_app::OptApp=FluxOptApp(),
        optimizer::Any=default_optimizer[typeof(opt_app)],
        n_epochs::Integer=128,
        adtype::SciMLBase.AbstractADType=GalacticOptim.AutoZygote(),

        batch_size::Integer=128,
        ) where {T <: AbstractFloat, T2 <: AbstractCondICNF{T}}
    CondICNFModel(m, loss, opt_app, optimizer, n_epochs, adtype, batch_size)
end

function MLJModelInterface.fit(model::CondICNFModel, verbosity, XY)
    X, Y = XY
    x = collect(MLJModelInterface.matrix(X)')
    y = collect(MLJModelInterface.matrix(Y)')
    data = DataLoader((x, y); batchsize=model.batch_size, shuffle=true, partial=true)
    ncdata = ncycle(data, model.n_epochs)
    initial_loss_value = model.loss(model.m, x, y)

    if model.opt_app isa FluxOptApp
        @assert model.optimizer isa Flux.Optimise.AbstractOptimiser
        _loss = loss_f(model.m, model.opt_app)
        _cb = cb_f(model.m, model.opt_app, model.loss, data)
        _p = Flux.params(model.m)
        t₀ = time()
        Flux.Optimise.train!(_loss, _p, ncdata, model.optimizer; cb=_cb)
        t₁ = time()
    elseif model.opt_app isa OptimOptApp
        @assert model.optimizer isa Optim.AbstractOptimizer
        itrtr = Any[nothing, nothing]
        itrtr .= iterate(ncdata)
        _loss = loss_f(model.m, model.opt_app, itrtr)
        _cb = cb_f(model.m, model.opt_app, model.loss, data, itrtr)
        ops = Optim.Options(
            x_abstol=-Inf, x_reltol=-Inf,
            f_abstol=-Inf, f_reltol=-Inf,
            g_abstol=-Inf, g_reltol=-Inf,
            outer_x_abstol=-Inf, outer_x_reltol=-Inf,
            outer_f_abstol=-Inf, outer_f_reltol=-Inf,
            outer_g_abstol=-Inf, outer_g_reltol=-Inf,
            f_calls_limit=0, g_calls_limit=0, h_calls_limit=0,
            allow_f_increases=true, allow_outer_f_increases=true,
            successive_f_tol=typemax(Int), iterations=typemax(Int), outer_iterations=typemax(Int),
            store_trace=false, trace_simplex=true, show_trace=false, extended_trace=true,
            show_every=1, callback=_cb, time_limit=Inf,
        )
        t₀ = time()
        res = optimize(_loss, model.m.p, model.optimizer, ops)
        t₁ = time()
        model.m.p .= res.minimizer
    elseif model.opt_app isa SciMLOptApp
        _loss = loss_f(model.m, model.opt_app)
        _cb = cb_f(model.m, model.opt_app, model.loss, data)
        optfunc = OptimizationFunction(_loss, model.adtype)
        optprob = OptimizationProblem(optfunc, model.m.p)
        t₀ = time()
        res = solve(optprob, model.optimizer, ncdata; cb=_cb)
        t₁ = time()
        model.m.p .= res.u
    end
    final_loss_value = model.loss(model.m, x, y)
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

function MLJModelInterface.transform(model::CondICNFModel, fitresult, XYnew)
    Xnew, Ynew = XYnew
    xnew = collect(MLJModelInterface.matrix(Xnew)')
    ynew = collect(MLJModelInterface.matrix(Ynew)')

    t₀ = time()
    logp̂x = inference(model.m, TestMode(), xnew, ynew)
    t₁ = time()
    Δt = t₁ - t₀
    @info "time cost (transform) = $(Δt) seconds"

    DataFrame(px=exp.(logp̂x))
end

function MLJModelInterface.fitted_params(model::CondICNFModel, fitresult)
    (
        learned_parameters=model.m.p,
    )
end

MLJBase.metadata_pkg.(
    [ICNFModel, CondICNFModel],
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
MLJBase.metadata_model(
    CondICNFModel,
    input_scitype=Tuple{Table{AbstractVector{ScientificTypes.Continuous}}, Table{AbstractVector{ScientificTypes.Continuous}}},
    target_scitype=Table{AbstractVector{ScientificTypes.Continuous}},
    output_scitype=Table{AbstractVector{ScientificTypes.Continuous}},
    supports_weights=false,
    docstring="CondICNFModel",
    load_path="ICNF.CondICNFModel",
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

export
    inference, generate,
    loss, loss_pn, loss_pln,
    loss_f, callback_f,
    CondICNFModel, CondICNFDist

function inference(icnf::AbstractCondICNF{T, AT}, mode::TestMode, xs::AbstractMatrix, ys::AbstractMatrix, p::AbstractVector=icnf.p)::AbstractVector where {T <: AbstractFloat, AT <: AbstractArray} end
function inference(icnf::AbstractCondICNF{T, AT}, mode::TrainMode, xs::AbstractMatrix, ys::AbstractMatrix, p::AbstractVector=icnf.p)::AbstractVector where {T <: AbstractFloat, AT <: AbstractArray} end

function generate(icnf::AbstractCondICNF{T, AT}, mode::TestMode, ys::AbstractMatrix, n::Integer, p::AbstractVector=icnf.p; rng::AbstractRNG=Random.default_rng())::AbstractMatrix where {T <: AbstractFloat, AT <: AbstractArray} end
function generate(icnf::AbstractCondICNF{T, AT}, mode::TrainMode, ys::AbstractMatrix, n::Integer, p::AbstractVector=icnf.p; rng::AbstractRNG=Random.default_rng())::AbstractMatrix where {T <: AbstractFloat, AT <: AbstractArray} end

function loss(icnf::AbstractCondICNF{T, AT}, xs::AbstractMatrix, ys::AbstractMatrix, p::AbstractVector=icnf.p; agg::Function=mean)::Number where {T <: AbstractFloat, AT <: AbstractArray} end

function loss_pn(icnf::AbstractCondICNF{T, AT}, xs::AbstractMatrix, ys::AbstractMatrix, p::AbstractVector=icnf.p; agg::Function=mean, nλ::T=convert(T, 1e-4))::Number where {T <: AbstractFloat, AT <: AbstractArray}
    lv = loss(icnf, xs, ys, p; agg)
    prm_n = norm(p)
    lv + nλ*prm_n
end

function loss_pln(icnf::AbstractCondICNF{T, AT}, xs::AbstractMatrix, ys::AbstractMatrix, p::AbstractVector=icnf.p; agg::Function=mean, nλ::T=convert(T, 1e-4))::Number where {T <: AbstractFloat, AT <: AbstractArray}
    lv = loss(icnf, xs, ys, p; agg)
    prm_ln = log(norm(p))
    lv + nλ*prm_ln
end

# -- Flux interface

function (icnf::AbstractCondICNF{T, AT})(xs::AbstractMatrix, ys::AbstractMatrix)::AbstractVector where {T <: AbstractFloat, AT <: AbstractArray}
    inference(icnf, TestMode(), xs, ys)
end

function loss_f(icnf::AbstractCondICNF{T, AT}, opt_app::FluxOptApp)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    function f(xs::AbstractMatrix, ys::AbstractMatrix)::T
        loss(icnf, xs, ys)
    end
    f
end

function callback_f(icnf::AbstractCondICNF{T, AT}, opt_app::FluxOptApp, loss::Function, data::DataLoader{T3})::Function where {T <: AbstractFloat, AT <: AbstractArray, T2 <: AbstractMatrix, T3 <: Tuple{T2, T2}}
    xs, ys = first(data)
    function f()::Nothing
        vl = loss(icnf, xs, ys)
        @info "Training" loss=vl
        nothing
    end
    f
end

# -- Optim interface

function loss_f(icnf::AbstractCondICNF{T, AT}, opt_app::OptimOptApp, itrtr::AbstractVector)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    function f(p::AbstractVector)::T
        xs, ys = itrtr[1]
        loss(icnf, xs, ys, p)
    end
    f
end

function callback_f(icnf::AbstractCondICNF{T, AT}, opt_app::OptimOptApp, loss::Function, data::DataLoader{T3}, itrtr::AbstractVector)::Function where {T <: AbstractFloat, AT <: AbstractArray, T2 <: AbstractMatrix, T3 <: Tuple{T2, T2}}
    xs, ys = first(data)
    function f(s::OptimizationState)::Bool
        vl = loss(icnf, xs, ys, s.metadata["x"])
        @info "Training" loss=vl
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

# -- SciML interface

function loss_f(icnf::AbstractCondICNF{T, AT}, opt_app::SciMLOptApp)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    function f(p::AbstractVector, θ::SciMLBase.NullParameters, xs::AbstractMatrix, ys::AbstractMatrix)
        loss(icnf, xs, ys, p)
    end
    f
end

function callback_f(icnf::AbstractCondICNF{T, AT}, opt_app::SciMLOptApp, loss::Function, data::DataLoader{T3})::Function where {T <: AbstractFloat, AT <: AbstractArray, T2 <: AbstractMatrix, T3 <: Tuple{T2, T2}}
    xs, ys = first(data)
    function f(p::AbstractVector, l::T)::Bool
        vl = loss(icnf, xs, ys, p)
        @info "Training" loss=vl
        false
    end
    f
end

# MLJ interface

mutable struct CondICNFModel <: MLJICNF
    m::AbstractCondICNF
    loss::Function

    opt_app::OptApp
    optimizer::Any
    n_epochs::Integer
    adtype::SciMLBase.AbstractADType

    batch_size::Integer

    array_type::UnionAll
end

function CondICNFModel(
        m::AbstractCondICNF,
        loss::Function=loss,
        ;
        opt_app::OptApp=FluxOptApp(),
        optimizer::Any=default_optimizer[typeof(opt_app)],
        n_epochs::Integer=128,
        adtype::SciMLBase.AbstractADType=Optimization.AutoZygote(),

        batch_size::Integer=128,
        )
    CondICNFModel(
        m, loss,
        opt_app, optimizer, n_epochs, adtype,
        batch_size,
        typeof(m).parameters[2],
    )
end

function MLJModelInterface.fit(model::CondICNFModel, verbosity, XY)
    X, Y = XY
    x = collect(transpose(MLJModelInterface.matrix(X)))
    x = convert(model.array_type, x)
    y = collect(transpose(MLJModelInterface.matrix(Y)))
    y = convert(model.array_type, y)
    data = DataLoader((x, y); batchsize=model.batch_size, shuffle=true, partial=true)
    ncdata = ncycle(data, model.n_epochs)
    initial_loss_value = model.loss(model.m, x, y)

    if model.opt_app isa FluxOptApp
        model.optimizer isa Flux.Optimise.AbstractOptimiser || error("model.optimizer must be a Flux optimizer")
        _loss = loss_f(model.m, model.opt_app)
        _callback = callback_f(model.m, model.opt_app, model.loss, data)
        _p = Flux.params(model.m)
        tst = @timed Flux.Optimise.train!(_loss, _p, ncdata, model.optimizer; cb=_callback)
    elseif model.opt_app isa OptimOptApp
        model.optimizer isa Optim.AbstractOptimizer || error("model.optimizer must be an Optim optimizer")
        itrtr = Any[nothing, nothing]
        itrtr .= iterate(ncdata)
        _loss = loss_f(model.m, model.opt_app, itrtr)
        _callback = callback_f(model.m, model.opt_app, model.loss, data, itrtr)
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
            show_every=1, callback=_callback, time_limit=Inf,
        )
        tst = @timed res = optimize(_loss, model.m.p, model.optimizer, ops)
        model.m.p .= res.minimizer
    elseif model.opt_app isa SciMLOptApp
        _loss = loss_f(model.m, model.opt_app)
        _callback = callback_f(model.m, model.opt_app, model.loss, data)
        optfunc = OptimizationFunction(_loss, model.adtype)
        optprob = OptimizationProblem(optfunc, model.m.p)
        tst = @timed res = solve(optprob, model.optimizer, ncdata; callback=_callback)
        model.m.p .= res.u
    end
    final_loss_value = model.loss(model.m, x, y)
    @info("Fitting",
        "elapsed time (seconds)"=tst.time,
        "garbage collection time (seconds)"=tst.gctime,
    )

    fitresult = nothing
    cache = nothing
    report = (
        stats=tst,
        initial_loss_value=initial_loss_value,
        final_loss_value=final_loss_value,
    )
    fitresult, cache, report
end

function MLJModelInterface.transform(model::CondICNFModel, fitresult, XYnew)
    Xnew, Ynew = XYnew
    xnew = collect(transpose(MLJModelInterface.matrix(Xnew)))
    xnew = convert(model.array_type, xnew)
    ynew = collect(transpose(MLJModelInterface.matrix(Ynew)))
    ynew = convert(model.array_type, ynew)

    tst = @timed logp̂x = inference(model.m, TestMode(), xnew, ynew)
    @info("Transforming",
        "elapsed time (seconds)"=tst.time,
        "garbage collection time (seconds)"=tst.gctime,
    )

    DataFrame(px=exp.(logp̂x))
end

function MLJModelInterface.fitted_params(model::CondICNFModel, fitresult)
    (
        learned_parameters=model.m.p,
    )
end

MLJBase.metadata_pkg(
    CondICNFModel,
    package_name="ICNF",
    package_uuid="9bd0f7d2-bd29-441d-bcde-0d11364d2762",
    package_url="https://github.com/impICNF/ICNF.jl",
    is_pure_julia=true,
    package_license="MIT",
    is_wrapper=false,
)
MLJBase.metadata_model(
    CondICNFModel,
    input_scitype=Tuple{Table{AbstractVector{ScientificTypes.Continuous}}, Table{AbstractVector{ScientificTypes.Continuous}}},
    target_scitype=Table{AbstractVector{ScientificTypes.Continuous}},
    output_scitype=Table{AbstractVector{ScientificTypes.Continuous}},
    supports_weights=false,
    load_path="ICNF.CondICNFModel",
)

# Distributions interface

struct CondICNFDist <: ICNFDistribution
    m::AbstractCondICNF
    ys::AbstractMatrix
end

Base.length(d::CondICNFDist) = d.m.nvars
Base.eltype(d::CondICNFDist) = typeof(d.m).parameters[1]
Distributions._logpdf(d::CondICNFDist, x::AbstractVector) = first(Distributions._logpdf(d, hcat(x)))
Distributions._logpdf(d::CondICNFDist, A::AbstractMatrix) = inference(d.m, TestMode(), A, d.ys[:, 1:size(A, 2)])
Distributions._rand!(rng::AbstractRNG, d::CondICNFDist, x::AbstractVector) = (x .= Distributions._rand!(rng, d, hcat(x)))
Distributions._rand!(rng::AbstractRNG, d::CondICNFDist, A::AbstractMatrix) = (A .= generate(d.m, TestMode(), d.ys[:, 1:size(A, 2)], size(A, 2); rng))

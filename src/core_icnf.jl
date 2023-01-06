export inference, generate, loss, loss_pn, loss_pln, loss_f, callback_f, ICNFModel, ICNFDist

function inference(
    icnf::AbstractICNF{T, AT},
    mode::TestMode,
    xs::AbstractMatrix,
    p::AbstractVector = icnf.p,
)::AbstractVector where {T <: AbstractFloat, AT <: AbstractArray} end
function inference(
    icnf::AbstractICNF{T, AT},
    mode::TrainMode,
    xs::AbstractMatrix,
    p::AbstractVector = icnf.p,
)::AbstractVector where {T <: AbstractFloat, AT <: AbstractArray} end

function generate(
    icnf::AbstractICNF{T, AT},
    mode::TestMode,
    n::Integer,
    p::AbstractVector = icnf.p;
    rng::AbstractRNG = Random.default_rng(),
)::AbstractMatrix where {T <: AbstractFloat, AT <: AbstractArray} end
function generate(
    icnf::AbstractICNF{T, AT},
    mode::TrainMode,
    n::Integer,
    p::AbstractVector = icnf.p;
    rng::AbstractRNG = Random.default_rng(),
)::AbstractMatrix where {T <: AbstractFloat, AT <: AbstractArray} end

function loss(
    icnf::AbstractICNF{T, AT},
    xs::AbstractMatrix,
    p::AbstractVector = icnf.p;
    agg::Function = mean,
)::Real where {T <: AbstractFloat, AT <: AbstractArray} end

function loss_pn(
    icnf::AbstractICNF{T, AT},
    xs::AbstractMatrix,
    p::AbstractVector = icnf.p;
    agg::Function = mean,
    nλ::T = convert(T, 1e-4),
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    lv = loss(icnf, xs, p; agg)
    prm_n = norm(p)
    lv + nλ * prm_n
end

function loss_pln(
    icnf::AbstractICNF{T, AT},
    xs::AbstractMatrix,
    p::AbstractVector = icnf.p;
    agg::Function = mean,
    nλ::T = convert(T, 1e-4),
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    lv = loss(icnf, xs, p; agg)
    prm_ln = log(norm(p))
    lv + nλ * prm_ln
end

# -- Flux interface

function (icnf::AbstractICNF{T, AT})(
    xs::AbstractMatrix,
)::AbstractVector where {T <: AbstractFloat, AT <: AbstractArray}
    inference(icnf, TestMode(), xs)
end

function loss_f(
    icnf::AbstractICNF{T, AT},
    opt_app::FluxOptApp,
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    loss
end

function callback_f(
    icnf::AbstractICNF{T, AT},
    opt_app::FluxOptApp,
    loss::Function,
    data::DataLoader{T3},
)::Function where {
    T <: AbstractFloat,
    AT <: AbstractArray,
    T2 <: AbstractMatrix,
    T3 <: Tuple{T2},
}
    xs, = first(data)
    function f()::Nothing
        vl = loss(icnf, xs)
        @info "Training" loss = vl
        nothing
    end
    f
end

# -- Optim interface

function loss_f(
    icnf::AbstractICNF{T, AT},
    opt_app::OptimOptApp,
    itrtr::AbstractVector,
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    function f(p::AbstractVector)::Real
        xs, = itrtr[1]
        loss(icnf, xs, p)
    end
    f
end

function callback_f(
    icnf::AbstractICNF{T, AT},
    opt_app::OptimOptApp,
    loss::Function,
    data::DataLoader{T3},
    itrtr::AbstractVector,
)::Function where {
    T <: AbstractFloat,
    AT <: AbstractArray,
    T2 <: AbstractMatrix,
    T3 <: Tuple{T2},
}
    xs, = first(data)
    function f(s::OptimizationState)::Bool
        vl = loss(icnf, xs, s.metadata["x"])
        @info "Training" loss = vl
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

function loss_f(
    icnf::AbstractICNF{T, AT},
    opt_app::SciMLOptApp,
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    function f(p::AbstractVector, θ::SciMLBase.NullParameters, xs::AbstractMatrix)::Real
        loss(icnf, xs, p)
    end
    f
end

function callback_f(
    icnf::AbstractICNF{T, AT},
    opt_app::SciMLOptApp,
    loss::Function,
    data::DataLoader{T3},
)::Function where {
    T <: AbstractFloat,
    AT <: AbstractArray,
    T2 <: AbstractMatrix,
    T3 <: Tuple{T2},
}
    xs, = first(data)
    function f(p::AbstractVector, l::T)::Bool
        vl = loss(icnf, xs, p)
        @info "Training" loss = vl
        false
    end
    f
end

# MLJ interface

mutable struct ICNFModel <: MLJICNF
    m::AbstractICNF
    loss::Function

    opt_app::OptApp
    optimizer::Any
    n_epochs::Integer
    adtype::SciMLBase.AbstractADType

    batch_size::Integer

    array_type::Type{<:AbstractArray}
end

function ICNFModel(
    m::AbstractICNF,
    loss::Function = loss,
    ;
    opt_app::OptApp = FluxOptApp(),
    optimizer::Any = default_optimizer[typeof(opt_app)],
    n_epochs::Integer = 128,
    adtype::SciMLBase.AbstractADType = Optimization.AutoZygote(),
    batch_size::Integer = 128,
)
    ICNFModel(
        m,
        loss,
        opt_app,
        optimizer,
        n_epochs,
        adtype,
        batch_size,
        typeof(m).parameters[2],
    )
end

function MLJModelInterface.fit(model::ICNFModel, verbosity, X)
    x = collect(transpose(MLJModelInterface.matrix(X)))
    x = convert(model.array_type, x)
    data = DataLoader((x,); batchsize = model.batch_size, shuffle = true, partial = true)
    ncdata = ncycle(data, model.n_epochs)
    initial_loss_value = model.loss(model.m, x)

    if model.opt_app isa FluxOptApp
        model.optimizer isa Optimisers.AbstractRule ||
            error("model.optimizer must be an Optimisers optimizer")
        _loss = loss_f(model.m, model.opt_app)
        _callback = callback_f(model.m, model.opt_app, model.loss, data)
        opt_state = Flux.setup(model.optimizer, model.m)
        tst =
            @timed Flux.train!(_loss, model.m, ncdata, opt_state)
    elseif model.opt_app isa OptimOptApp
        model.optimizer isa Optim.AbstractOptimizer ||
            error("model.optimizer must be an Optim optimizer")
        itrtr = Any[nothing, nothing]
        itrtr .= iterate(ncdata)
        _loss = loss_f(model.m, model.opt_app, itrtr)
        _callback = callback_f(model.m, model.opt_app, model.loss, data, itrtr)
        ops = Optim.Options(;
            x_abstol = -Inf,
            x_reltol = -Inf,
            f_abstol = -Inf,
            f_reltol = -Inf,
            g_abstol = -Inf,
            g_reltol = -Inf,
            outer_x_abstol = -Inf,
            outer_x_reltol = -Inf,
            outer_f_abstol = -Inf,
            outer_f_reltol = -Inf,
            outer_g_abstol = -Inf,
            outer_g_reltol = -Inf,
            f_calls_limit = 0,
            g_calls_limit = 0,
            h_calls_limit = 0,
            allow_f_increases = true,
            allow_outer_f_increases = true,
            successive_f_tol = typemax(Int),
            iterations = typemax(Int),
            outer_iterations = typemax(Int),
            store_trace = false,
            trace_simplex = true,
            show_trace = false,
            extended_trace = true,
            show_every = 1,
            callback = _callback,
            time_limit = Inf,
        )
        tst = @timed res = optimize(_loss, model.m.p, model.optimizer, ops)
        model.m.p .= res.minimizer
    elseif model.opt_app isa SciMLOptApp
        _loss = loss_f(model.m, model.opt_app)
        _callback = callback_f(model.m, model.opt_app, model.loss, data)
        optfunc = OptimizationFunction(_loss, model.adtype)
        optprob = OptimizationProblem(optfunc, model.m.p)
        tst = @timed res = solve(optprob, model.optimizer, ncdata; callback = _callback)
        model.m.p .= res.u
    end
    final_loss_value = model.loss(model.m, x)
    @info(
        "Fitting",
        "elapsed time (seconds)" = tst.time,
        "garbage collection time (seconds)" = tst.gctime,
    )

    fitresult = nothing
    cache = nothing
    report = (
        stats = tst,
        initial_loss_value = initial_loss_value,
        final_loss_value = final_loss_value,
    )
    fitresult, cache, report
end

function MLJModelInterface.transform(model::ICNFModel, fitresult, Xnew)
    xnew = collect(transpose(MLJModelInterface.matrix(Xnew)))
    xnew = convert(model.array_type, xnew)

    tst = @timed logp̂x = inference(model.m, TestMode(), xnew)
    @info(
        "Transforming",
        "elapsed time (seconds)" = tst.time,
        "garbage collection time (seconds)" = tst.gctime,
    )

    DataFrame(; px = exp.(logp̂x))
end

function MLJModelInterface.fitted_params(model::ICNFModel, fitresult)
    (learned_parameters = model.m.p,)
end

MLJBase.metadata_pkg(
    ICNFModel;
    package_name = "ICNF",
    package_uuid = "9bd0f7d2-bd29-441d-bcde-0d11364d2762",
    package_url = "https://github.com/impICNF/ICNF.jl",
    is_pure_julia = true,
    package_license = "MIT",
    is_wrapper = false,
)
MLJBase.metadata_model(
    ICNFModel;
    input_scitype = Table{AbstractVector{ScientificTypes.Continuous}},
    target_scitype = Table{AbstractVector{ScientificTypes.Continuous}},
    output_scitype = Table{AbstractVector{ScientificTypes.Continuous}},
    supports_weights = false,
    load_path = "ICNF.ICNFModel",
)

# Distributions interface

struct ICNFDist <: ICNFDistribution
    m::AbstractICNF
end

Base.length(d::ICNFDist) = d.m.nvars
Base.eltype(d::ICNFDist) = typeof(d.m).parameters[1]
function Distributions._logpdf(d::ICNFDist, x::AbstractVector)
    first(Distributions._logpdf(d, hcat(x)))
end
Distributions._logpdf(d::ICNFDist, A::AbstractMatrix) = inference(d.m, TestMode(), A)
function Distributions._rand!(rng::AbstractRNG, d::ICNFDist, x::AbstractVector)
    (x .= Distributions._rand!(rng, d, hcat(x)))
end
function Distributions._rand!(rng::AbstractRNG, d::ICNFDist, A::AbstractMatrix)
    (A .= generate(d.m, TestMode(), size(A, 2); rng))
end

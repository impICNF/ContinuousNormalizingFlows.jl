export loss_f, callback_f, CondICNFModel, CondICNFDist

# -- SciML interface

function loss_f(
    icnf::AbstractCondICNF{T, AT},
    loss::Function,
    st::Any,
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    function f(ps, θ, xs, ys)
        loss(icnf, xs, ys, ps, st)
    end
    f
end

function callback_f(
    icnf::AbstractCondICNF{T, AT},
    loss::Function,
    data::DataLoader{T3},
    st::Any,
)::Function where {
    T <: AbstractFloat,
    AT <: AbstractArray,
    T2 <: AbstractMatrix{<:Real},
    T3 <: Tuple{T2, T2},
}
    xs, ys = first(data)
    function f(ps, l)
        vl = loss(icnf, xs, ys, ps, st)
        @info "Training" loss = vl
        false
    end
    f
end

# MLJ interface

mutable struct CondICNFModel <: MLJICNF
    m::AbstractCondICNF
    loss::Function

    optimizer::Any
    n_epochs::Integer
    adtype::SciMLBase.AbstractADType

    batch_size::Integer

    array_type::Type{<:AbstractArray}
end

function CondICNFModel(
    m::AbstractCondICNF,
    loss::Function = loss,
    ;
    optimizer::Any = default_optimizer,
    n_epochs::Integer = 128,
    adtype::SciMLBase.AbstractADType = Optimization.AutoZygote(),
    batch_size::Integer = 128,
)
    CondICNFModel(m, loss, optimizer, n_epochs, adtype, batch_size, typeof(m).parameters[2])
end

function MLJModelInterface.fit(model::CondICNFModel, verbosity, XY)
    rng = Random.default_rng()
    X, Y = XY
    x = collect(transpose(MLJModelInterface.matrix(X)))
    x = convert(model.array_type, x)
    y = collect(transpose(MLJModelInterface.matrix(Y)))
    y = convert(model.array_type, y)
    data = DataLoader((x, y); batchsize = model.batch_size, shuffle = true, partial = true)
    ncdata = ncycle(data, model.n_epochs)
    ps, st = LuxCore.setup(rng, model.m)
    initial_loss_value = model.loss(model.m, first(data)..., ps, st)
    _loss = loss_f(model.m, model.loss, st)
    _callback = callback_f(model.m, model.loss, data, st)
    optfunc = OptimizationFunction(_loss, model.adtype)
    optprob = OptimizationProblem(optfunc, ps)
    tst = @timed res = solve(optprob, model.optimizer, ncdata; callback = _callback)
    ps .= res.u
    final_loss_value = model.loss(model.m, first(data)..., ps, st)
    @info(
        "Fitting",
        "elapsed time (seconds)" = tst.time,
        "garbage collection time (seconds)" = tst.gctime,
    )

    fitresult = (ps, st)
    cache = nothing
    report = (
        stats = tst,
        initial_loss_value = initial_loss_value,
        final_loss_value = final_loss_value,
    )
    fitresult, cache, report
end

function MLJModelInterface.transform(model::CondICNFModel, fitresult, XYnew)
    Xnew, Ynew = XYnew
    xnew = collect(transpose(MLJModelInterface.matrix(Xnew)))
    xnew = convert(model.array_type, xnew)
    ynew = collect(transpose(MLJModelInterface.matrix(Ynew)))
    ynew = convert(model.array_type, ynew)
    (ps, st) = fitresult

    tst = @timed logp̂x = Folds.map(
        ((x, y),) -> first(inference(model.m, TestMode(), x, y, ps, st)),
        zip(eachcol(xnew), eachcol(ynew)),
    )
    @info(
        "Transforming",
        "elapsed time (seconds)" = tst.time,
        "garbage collection time (seconds)" = tst.gctime,
    )

    DataFrame(; px = exp.(logp̂x))
end

function MLJModelInterface.fitted_params(model::CondICNFModel, fitresult)
    (ps, st) = fitresult
    (learned_parameters = ps, states = st)
end

MLJBase.metadata_pkg(
    CondICNFModel;
    package_name = "ICNF",
    package_uuid = "9bd0f7d2-bd29-441d-bcde-0d11364d2762",
    package_url = "https://github.com/impICNF/ICNF.jl",
    is_pure_julia = true,
    package_license = "MIT",
    is_wrapper = false,
)
MLJBase.metadata_model(
    CondICNFModel;
    input_scitype = Tuple{
        Table{AbstractVector{ScientificTypes.Continuous}},
        Table{AbstractVector{ScientificTypes.Continuous}},
    },
    target_scitype = Table{AbstractVector{ScientificTypes.Continuous}},
    output_scitype = Table{AbstractVector{ScientificTypes.Continuous}},
    supports_weights = false,
    load_path = "ICNF.CondICNFModel",
)

# Distributions interface

struct CondICNFDist <: ICNFDistribution
    m::AbstractCondICNF
    ys::AbstractVector{<:Real}
    ps::Any
    st::Any
end

Base.length(d::CondICNFDist) = d.m.nvars
Base.eltype(d::CondICNFDist) = typeof(d.m).parameters[1]
function Distributions._logpdf(d::CondICNFDist, x::AbstractVector{<:Real})
    first(inference(d.m, TestMode(), x, d.ys, d.ps, d.st))
end
function Distributions._logpdf(d::CondICNFDist, A::AbstractMatrix{<:Real})
    Folds.map(x -> Distributions._logpdf(d, x), eachcol(A))
end
function Distributions._rand!(rng::AbstractRNG, d::CondICNFDist, x::AbstractVector{<:Real})
    x .= generate(d.m, TestMode(), d.ys, d.ps, d.st; rng)
end
function Distributions._rand!(rng::AbstractRNG, d::CondICNFDist, A::AbstractMatrix{<:Real})
    A .= hcat(Folds.map(x -> Distributions._rand!(rng, d, x), eachcol(A))...)
end

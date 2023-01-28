export loss_f, callback_f, ICNFModel, ICNFDist

# -- Flux interface

function (icnf::AbstractICNF{T, AT})(
    xs::AbstractVector{<:Real},
)::Real where {T <: AbstractFloat, AT <: AbstractArray}
    first(inference(icnf, TestMode(), xs))
end

function (icnf::AbstractICNF{T, AT})(
    xs::AbstractMatrix{<:Real},
)::AbstractVector{<:Real} where {T <: AbstractFloat, AT <: AbstractArray}
    Folds.map(x -> first(inference(icnf, TestMode(), x)), eachcol(xs))
end

# -- SciML interface

function loss_f(
    icnf::AbstractICNF{T, AT},
    loss::Function,
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    function f(
        p::AbstractVector{<:Real},
        θ::SciMLBase.NullParameters,
        xs::AbstractMatrix{<:Real},
    )::Real
        loss(icnf, xs, p)
    end
    f
end

function callback_f(
    icnf::AbstractICNF{T, AT},
    loss::Function,
    data::DataLoader{T3},
)::Function where {
    T <: AbstractFloat,
    AT <: AbstractArray,
    T2 <: AbstractMatrix{<:Real},
    T3 <: Tuple{T2},
}
    xs, = first(data)
    function f(p::AbstractVector{<:Real}, l::Real)::Bool
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
    optimizer::Any = default_optimizer,
    n_epochs::Integer = 128,
    adtype::SciMLBase.AbstractADType = Optimization.AutoZygote(),
    batch_size::Integer = 128,
)
    ICNFModel(m, loss, optimizer, n_epochs, adtype, batch_size, typeof(m).parameters[2])
end

function MLJModelInterface.fit(model::ICNFModel, verbosity, X)
    x = collect(transpose(MLJModelInterface.matrix(X)))
    x = convert(model.array_type, x)
    data = DataLoader((x,); batchsize = model.batch_size, shuffle = true, partial = true)
    ncdata = ncycle(data, model.n_epochs)
    initial_loss_value = model.loss(model.m, first(data)...)
    _loss = loss_f(model.m, model.loss)
    _callback = callback_f(model.m, model.loss, data)
    optfunc = OptimizationFunction(_loss, model.adtype)
    optprob = OptimizationProblem(optfunc, model.m.p)
    tst = @timed res = solve(optprob, model.optimizer, ncdata; callback = _callback)
    model.m.p .= res.u
    final_loss_value = model.loss(model.m, first(data)...)
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

    tst = @timed logp̂x = Folds.map(x -> first(inference(model.m, TestMode(), x)), eachcol(xnew))
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
function Distributions._logpdf(d::ICNFDist, x::AbstractVector{<:Real})
    first(inference(d.m, TestMode(), x))
end
function Distributions._logpdf(d::ICNFDist, A::AbstractMatrix{<:Real})
    Folds.map(x -> Distributions._logpdf(d, x), eachcol(A))
end
function Distributions._rand!(rng::AbstractRNG, d::ICNFDist, x::AbstractVector{<:Real})
    x .= generate(d.m, TestMode(); rng)
end
function Distributions._rand!(rng::AbstractRNG, d::ICNFDist, A::AbstractMatrix{<:Real})
    A .= Folds.map(x -> Distributions._rand!(rng, d, x), eachcol(A))
end

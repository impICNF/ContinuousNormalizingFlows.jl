mutable struct CondICNFModel{AICNF <: AbstractICNF} <: MLJICNF{AICNF}
    icnf::AICNF
    loss::Function
    optimizers::Tuple
    batchsize::Int
    adtype::ADTypes.AbstractADType
    sol_kwargs::NamedTuple
end

function CondICNFModel(;
    icnf::AbstractICNF = ICNF(),
    loss::Function = loss,
    optimizers::Tuple = (
        Optimisers.OptimiserChain(Optimisers.WeightDecay(), Optimisers.Adam()),
    ),
    batchsize::Int = 1024,
    adtype::ADTypes.AbstractADType = ADTypes.AutoZygote(),
    sol_kwargs::NamedTuple = (; epochs = 300, progress = true),
)
    return CondICNFModel(icnf, loss, optimizers, batchsize, adtype, sol_kwargs)
end

function MLJModelInterface.fit(model::CondICNFModel, verbosity, XY)
    X, Y = XY
    x = collect(transpose(MLJModelInterface.matrix(X)))
    y = collect(transpose(MLJModelInterface.matrix(Y)))
    ps, st = LuxCore.setup(model.icnf.rng, model.icnf)
    ps = ComponentArrays.ComponentArray(ps)
    x = model.icnf.device(x)
    y = model.icnf.device(y)
    ps = model.icnf.device(ps)
    st = model.icnf.device(st)
    data = make_dataloader(model.icnf, model.batchsize, (x, y))
    data = model.icnf.device(data)
    optprob = SciMLBase.OptimizationProblem{true}(
        SciMLBase.OptimizationFunction{true}(
            make_opt_loss(model.icnf, TrainMode{true}(), st, model.loss),
            model.adtype,
        ),
        ps,
        data,
    )
    res_stats = Any[]
    for opt in model.optimizers
        optprob_re = SciMLBase.remake(optprob; u0 = ps)
        res = SciMLBase.solve(optprob_re, opt; model.sol_kwargs...)
        ps .= res.u
        push!(res_stats, res.stats)
    end

    fitresult = (ps, st)
    cache = nothing
    report = (stats = res_stats,)
    return (fitresult, cache, report)
end

function MLJModelInterface.transform(model::CondICNFModel, fitresult, XYnew)
    Xnew, Ynew = XYnew
    xnew = collect(transpose(MLJModelInterface.matrix(Xnew)))
    ynew = collect(transpose(MLJModelInterface.matrix(Ynew)))
    xnew = model.icnf.device(xnew)
    ynew = model.icnf.device(ynew)
    (ps, st) = fitresult

    logp̂x = if model.icnf.compute_mode isa VectorMode
        @warn "to compute by vectors, data should be a vector." maxlog = 1
        broadcast(
            function (x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
                return first(inference(model.icnf, TestMode{false}(), x, y, ps, st))
            end,
            collect(collect.(eachcol(xnew))),
            collect(collect.(eachcol(ynew))),
        )
    elseif model.icnf.compute_mode isa MatrixMode
        first(inference(model.icnf, TestMode{false}(), xnew, ynew, ps, st))
    else
        error("Not Implemented")
    end

    return DataFrames.DataFrame(; px = exp.(logp̂x))
end

MLJBase.metadata_pkg(
    CondICNFModel;
    package_name = "ContinuousNormalizingFlows",
    package_uuid = "00b1973d-5b2e-40bf-8604-5c9c1d8f50ac",
    package_url = "https://github.com/impICNF/ContinuousNormalizingFlows.jl",
    is_pure_julia = true,
    package_license = "MIT",
    is_wrapper = false,
)
MLJBase.metadata_model(
    CondICNFModel;
    input_scitype = Tuple{
        ScientificTypesBase.Table{AbstractVector{ScientificTypesBase.Continuous}},
        ScientificTypesBase.Table{AbstractVector{ScientificTypesBase.Continuous}},
    },
    target_scitype = ScientificTypesBase.Table{
        AbstractVector{ScientificTypesBase.Continuous},
    },
    output_scitype = ScientificTypesBase.Table{
        AbstractVector{ScientificTypesBase.Continuous},
    },
    supports_weights = false,
    load_path = "ContinuousNormalizingFlows.CondICNFModel",
)

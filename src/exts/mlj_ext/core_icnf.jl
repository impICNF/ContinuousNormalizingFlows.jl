mutable struct ICNFModel{AICNF <: AbstractICNF} <: MLJICNF{AICNF}
    m::AICNF
    loss::Function

    optimizers::Tuple
    adtype::ADTypes.AbstractADType

    batchsize::Int
    sol_kwargs::NamedTuple
end

function ICNFModel(
    m::AbstractICNF,
    loss::Function = loss;
    optimizers::Tuple = (Optimisers.Adam(),),
    adtype::ADTypes.AbstractADType = ADTypes.AutoZygote(),
    batchsize::Int = 32,
    sol_kwargs::NamedTuple = (;),
)
    return ICNFModel(m, loss, optimizers, adtype, batchsize, sol_kwargs)
end

function MLJModelInterface.fit(model::ICNFModel, verbosity, X)
    x = collect(transpose(MLJModelInterface.matrix(X)))
    ps, st = LuxCore.setup(model.m.rng, model.m)
    ps = ComponentArrays.ComponentArray(ps)
    x = model.m.device(x)
    ps = model.m.device(ps)
    st = model.m.device(st)
    data = make_dataloader(model.m, model.batchsize, (x,))
    data = model.m.device(data)
    optprob = SciMLBase.OptimizationProblem{true}(
        SciMLBase.OptimizationFunction{true}(
            make_opt_loss(model.m, TrainMode{true}(), st, model.loss),
            model.adtype,
        ),
        ps,
        data,
    )
    res_stats = SciMLBase.OptimizationStats[]
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

function MLJModelInterface.transform(model::ICNFModel, fitresult, Xnew)
    xnew = collect(transpose(MLJModelInterface.matrix(Xnew)))
    xnew = model.m.device(xnew)
    (ps, st) = fitresult

    logp̂x = if model.m.compute_mode isa VectorMode
        @warn maxlog = 1 "to compute by vectors, data should be a vector."
        broadcast(
            function (x::AbstractVector{<:Real})
                return first(inference(model.m, TestMode{false}(), x, ps, st))
            end,
            collect(collect.(eachcol(xnew))),
        )
    elseif model.m.compute_mode isa MatrixMode
        first(inference(model.m, TestMode{false}(), xnew, ps, st))
    else
        error("Not Implemented")
    end

    return DataFrames.DataFrame(; px = exp.(logp̂x))
end

MLJBase.metadata_pkg(
    ICNFModel;
    package_name = "ContinuousNormalizingFlows",
    package_uuid = "00b1973d-5b2e-40bf-8604-5c9c1d8f50ac",
    package_url = "https://github.com/impICNF/ContinuousNormalizingFlows.jl",
    is_pure_julia = true,
    package_license = "MIT",
    is_wrapper = false,
)
MLJBase.metadata_model(
    ICNFModel;
    input_scitype = ScientificTypesBase.Table{
        AbstractVector{ScientificTypesBase.Continuous},
    },
    target_scitype = ScientificTypesBase.Table{
        AbstractVector{ScientificTypesBase.Continuous},
    },
    output_scitype = ScientificTypesBase.Table{
        AbstractVector{ScientificTypesBase.Continuous},
    },
    supports_weights = false,
    load_path = "ContinuousNormalizingFlows.ICNFModel",
)

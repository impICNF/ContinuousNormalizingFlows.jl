mutable struct CondICNFModel{AICNF <: AbstractICNF} <: MLJICNF{AICNF}
    m::AICNF
    loss::Function

    optimizers::Tuple
    n_epochs::Int
    adtype::ADTypes.AbstractADType

    batch_size::Int
    sol_kwargs::NamedTuple
end

function CondICNFModel(
    m::AbstractICNF,
    loss::Function = loss;
    optimizers::Tuple = (Optimisers.Lion(),),
    n_epochs::Int = 300,
    adtype::ADTypes.AbstractADType = ADTypes.AutoZygote(),
    batch_size::Int = 32,
    sol_kwargs::NamedTuple = (;),
)
    return CondICNFModel(m, loss, optimizers, n_epochs, adtype, batch_size, sol_kwargs)
end

function MLJModelInterface.fit(model::CondICNFModel, verbosity, XY)
    X, Y = XY
    x = collect(transpose(MLJModelInterface.matrix(X)))
    y = collect(transpose(MLJModelInterface.matrix(Y)))
    ps, st = LuxCore.setup(model.m.rng, model.m)
    ps = ComponentArrays.ComponentArray(ps)
    x = model.m.device(x)
    y = model.m.device(y)
    ps = model.m.device(ps)
    st = model.m.device(st)
    data = if model.m.compute_mode isa VectorMode
        MLUtils.DataLoader((x, y); batchsize = -1, shuffle = true, partial = true)
    elseif model.m.compute_mode isa MatrixMode
        MLUtils.DataLoader(
            (x, y);
            batchsize = if iszero(model.batch_size)
                max(size(x, 2), size(y, 2))
            else
                model.batch_size
            end,
            shuffle = true,
            partial = true,
        )
    else
        error("Not Implemented")
    end
    data = model.m.device(data)
    optfunc = SciMLBase.OptimizationFunction(
        make_opt_loss(model.m, TrainMode(), st, model.loss),
        model.adtype,
    )
    optprob = SciMLBase.OptimizationProblem(optfunc, ps, data)
    res_stats = []
    for opt in model.optimizers
        optprob_re = SciMLBase.remake(optprob; u0 = ps)
        res = SciMLBase.solve(optprob_re, opt; epochs = model.n_epochs, model.sol_kwargs...)
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
    xnew = model.m.device(xnew)
    ynew = model.m.device(ynew)
    (ps, st) = fitresult

    logp̂x = if model.m.compute_mode isa VectorMode
        broadcast(
            function (x, y)
                return first(inference(model.m, TestMode(), x, y, ps, st))
            end,
            eachcol(xnew),
            eachcol(ynew),
        )
    elseif model.m.compute_mode isa MatrixMode
        first(inference(model.m, TestMode(), xnew, ynew, ps, st))
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

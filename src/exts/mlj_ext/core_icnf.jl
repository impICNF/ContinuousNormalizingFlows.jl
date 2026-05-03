mutable struct ICNFModel{AICNF <: AbstractICNF} <: MLJICNF{AICNF}
    icnf::AICNF
    loss::Function
    optimizers::Tuple
    batchsize::Int
    adtype::ADTypes.AbstractADType
    sol_kwargs::NamedTuple
end

function ICNFModel(;
    icnf::AbstractICNF = ICNF(),
    loss::Function = loss,
    optimizers::Tuple = (
        Optimisers.OptimiserChain(
            Optimisers.ClipNorm(
                one(eltype(icnf)),
                convert(eltype(icnf), 2.0e0);
                throw = true,
            ),
            Optimisers.WeightDecay(; lambda = convert(eltype(icnf), 1.0e-2)),
            Optimisers.Adam(;
                eta = convert(eltype(icnf), 1.0e-3),
                beta = (convert(eltype(icnf), 9.0e-1), convert(eltype(icnf), 9.99e-1)),
                epsilon = eps(eltype(icnf)),
            ),
        ),
    ),
    batchsize::Int = 1024,
    adtype::ADTypes.AbstractADType = ADTypes.AutoZygote(),
    sol_kwargs::NamedTuple = (;
        epochs = 300,
        progress = true,
        callback = make_opt_callback(64),
        verbose = OptimizationBase.OptimizationVerbosity(SciMLLogging.Detailed()),
    ),
)
    return ICNFModel(icnf, loss, optimizers, batchsize, adtype, sol_kwargs)
end

function MLJModelInterface.fit(model::ICNFModel, verbosity, X)
    x = permutedims(MLJModelInterface.matrix(X))
    ps, st = LuxCore.setup(model.icnf.rng, model.icnf)
    ps = ComponentArrays.ComponentArray(ps)
    eltype_adaptor = Lux.LuxEltypeAdaptor{eltype(model.icnf)}()
    x = model.icnf.device(eltype_adaptor(x))
    ps = model.icnf.device(eltype_adaptor(ps))
    st = model.icnf.device(eltype_adaptor(st))
    data = make_dataloader(model.icnf, model.batchsize, (x,))
    data = model.icnf.device(data)
    optprob = SciMLBase.OptimizationProblem{true}(
        SciMLBase.OptimizationFunction{true}(
            make_opt_loss(model.icnf, TrainMode{true}(), st, model.loss),
            model.adtype,
        ),
        ps,
        data;
        model.sol_kwargs...,
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

function MLJModelInterface.transform(model::ICNFModel, fitresult, Xnew)
    xnew = permutedims(MLJModelInterface.matrix(Xnew))
    eltype_adaptor = Lux.LuxEltypeAdaptor{eltype(model.icnf)}()
    xnew = model.icnf.device(eltype_adaptor(xnew))
    (ps, st) = fitresult

    logp̂x = get_logp̂x(model.icnf, xnew, ps, st)
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

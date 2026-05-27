mutable struct CondICNFModel{AICNF <: AbstractICNF} <: MLJICNF{AICNF}
    icnf::AICNF
    loss::Function
    batchsize::Int
    adtype::ADTypes.AbstractADType
    sol_kwargs::NamedTuple
end

function CondICNFModel(;
    icnf::AbstractICNF = ICNF(),
    loss::Function = loss,
    batchsize::Int = 1024,
    adtype::ADTypes.AbstractADType = ADTypes.AutoZygote(),
    sol_kwargs::NamedTuple = (;
        epochs = 300,
        callback = make_opt_callback(64),
        alg = Optimisers.OptimiserChain(
            Optimisers.WeightDecay(; lambda = convert(eltype(icnf), 1.0e-4)),
            Optimisers.ClipNorm(
                convert(eltype(icnf), 10.0),
                convert(eltype(icnf), 2.0);
                throw = true,
            ),
            Optimisers.Adam(;
                eta = convert(eltype(icnf), 0.001),
                beta = (convert(eltype(icnf), 0.9), convert(eltype(icnf), 0.999)),
                epsilon = convert(eltype(icnf), 1.0e-8),
            ),
        ),
        progress = true,
        verbose = SciMLLogging.Detailed(),
    ),
)
    return CondICNFModel(icnf, loss, batchsize, adtype, sol_kwargs)
end

function MLJModelInterface.fit(model::CondICNFModel, verbosity, XY)
    X, Y = XY
    x = permutedims(MLJModelInterface.matrix(X))
    y = permutedims(MLJModelInterface.matrix(Y))
    ps, st = LuxCore.setup(model.icnf.rng, model.icnf)
    ps = ComponentArrays.ComponentArray(ps)
    eltype_adaptor = Lux.LuxEltypeAdaptor{eltype(model.icnf)}()
    x = model.icnf.device(eltype_adaptor(x))
    y = model.icnf.device(eltype_adaptor(y))
    ps = model.icnf.device(eltype_adaptor(ps))
    st = model.icnf.device(eltype_adaptor(st))
    data = make_dataloader(model.icnf, model.batchsize, (x, y))
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
    res = SciMLBase.solve(optprob; model.sol_kwargs...)
    ps .= res.u

    fitresult = (ps, st)
    cache = nothing
    report = (stats = res.stats,)
    return (fitresult, cache, report)
end

function MLJModelInterface.transform(model::CondICNFModel, fitresult, (Xnew, Ynew))
    xnew = permutedims(MLJModelInterface.matrix(Xnew))
    ynew = permutedims(MLJModelInterface.matrix(Ynew))
    eltype_adaptor = Lux.LuxEltypeAdaptor{eltype(model.icnf)}()
    xnew = model.icnf.device(eltype_adaptor(xnew))
    ynew = model.icnf.device(eltype_adaptor(ynew))
    (ps, st) = fitresult

    logp̂x = get_logp̂x(model.icnf, xnew, ynew, ps, st)

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

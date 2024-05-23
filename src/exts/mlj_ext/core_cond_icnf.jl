mutable struct CondICNFModel{AICNF <: AbstractICNF} <: MLJICNF{AICNF}
    m::AICNF
    loss::Function

    optimizers::Tuple
    n_epochs::Int
    adtype::ADTypes.AbstractADType

    use_batch::Bool
    batch_size::Int
end

function CondICNFModel(
    m::AbstractICNF,
    loss::Function = loss;
    optimizers::Tuple = (Optimisers.Lion(),),
    n_epochs::Int = 300,
    adtype::ADTypes.AbstractADType = ADTypes.AutoZygote(),
    use_batch::Bool = true,
    batch_size::Int = 32,
)
    CondICNFModel(m, loss, optimizers, n_epochs, adtype, use_batch, batch_size)
end

function MLJModelInterface.fit(model::CondICNFModel, verbosity, XY)
    X, Y = XY
    x = collect(transpose(MLJModelInterface.matrix(X)))
    y = collect(transpose(MLJModelInterface.matrix(Y)))
    ps, st = LuxCore.setup(model.m.rng, model.m)
    ps = ComponentArrays.ComponentArray(ps)
    if model.m.resource isa ComputationalResources.CUDALibs
        gdev = Lux.gpu_device()
        x = gdev(x)
        y = gdev(y)
        ps = gdev(ps)
        st = gdev(st)
    end
    optfunc = SciMLBase.OptimizationFunction(
        make_opt_loss(model.m, TrainMode(), st, model.loss),
        model.adtype,
    )
    optprob = SciMLBase.OptimizationProblem(optfunc, ps)
    tst_overall = @timed for opt in model.optimizers
        tst_epochs = @timed for ep in 1:(model.n_epochs)
            if model.use_batch
                if model.m.compute_mode isa VectorMode
                    data = MLUtils.DataLoader(
                        (x, y);
                        batchsize = -1,
                        shuffle = true,
                        partial = true,
                        parallel = false,
                        buffer = false,
                    )
                elseif model.m.compute_mode isa MatrixMode
                    data = MLUtils.DataLoader(
                        (x, y);
                        batchsize = model.batch_size,
                        shuffle = true,
                        partial = true,
                        parallel = false,
                        buffer = false,
                    )
                else
                    error("Not Implemented")
                end
            else
                data = [(x, y)]
            end
            optprob_re = SciMLBase.remake(optprob; u0 = ps)
            tst_one =
                @timed res = SciMLBase.solve(optprob_re, opt, data; progress = true)
            ps .= res.u
            @info(
                "Fitting (epoch: $ep of $(model.n_epochs)) - $(typeof(opt).name.name)",
                "elapsed time (seconds)" = tst_one.time,
                "garbage collection time (seconds)" = tst_one.gctime,
                "allocated (bytes)" = tst_one.bytes,
                "final loss value" = res.objective,
            )
        end
        @info(
            "Fitting (all epochs) - $(typeof(opt).name.name)",
            "elapsed time (seconds)" = tst_epochs.time,
            "garbage collection time (seconds)" = tst_epochs.gctime,
            "allocated (bytes)" = tst_epochs.bytes,
        )
    end
    @info(
        "Fitting - Overall",
        "elapsed time (seconds)" = tst_overall.time,
        "garbage collection time (seconds)" = tst_overall.gctime,
        "allocated (bytes)" = tst_overall.bytes,
    )

    fitresult = (ps, st)
    cache = nothing
    report = (stats = tst_overall,)
    (fitresult, cache, report)
end

function MLJModelInterface.transform(model::CondICNFModel, fitresult, XYnew)
    Xnew, Ynew = XYnew
    xnew = collect(transpose(MLJModelInterface.matrix(Xnew)))
    ynew = collect(transpose(MLJModelInterface.matrix(Ynew)))
    if model.m.resource isa ComputationalResources.CUDALibs
        gdev = Lux.gpu_device()
        xnew = gdev(xnew)
        ynew = gdev(ynew)
    end
    (ps, st) = fitresult

    tst = @timed if model.m.compute_mode isa VectorMode
        logp̂x = broadcast(
            (x, y) -> first(inference(model.m, TestMode(), x, y, ps, st)),
            eachcol(xnew),
            eachcol(ynew),
        )
    elseif model.m.compute_mode isa MatrixMode
        logp̂x = first(inference(model.m, TestMode(), xnew, ynew, ps, st))
    else
        error("Not Implemented")
    end
    @info(
        "Transforming",
        "elapsed time (seconds)" = tst.time,
        "garbage collection time (seconds)" = tst.gctime,
        "allocated (bytes)" = tst.bytes,
    )

    DataFrames.DataFrame(; px = exp.(logp̂x))
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

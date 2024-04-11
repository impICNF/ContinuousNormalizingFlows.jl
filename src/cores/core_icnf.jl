export ICNFModel, ICNFDist

# MLJ interface

mutable struct ICNFModel <: MLJICNF
    m::AbstractICNF
    loss::Function

    optimizers::Tuple
    n_epochs::Int
    adtype::ADTypes.AbstractADType

    use_batch::Bool
    batch_size::Int

    compute_mode::Type{<:ComputeMode}
end

function ICNFModel(
    m::AbstractICNF{<:AbstractFloat, CM},
    loss::Function = loss;
    optimizers::Tuple = (Lion(),),
    n_epochs::Int = 300,
    adtype::ADTypes.AbstractADType = AutoZygote(),
    use_batch::Bool = true,
    batch_size::Int = 32,
) where {CM <: ComputeMode}
    ICNFModel(m, loss, optimizers, n_epochs, adtype, use_batch, batch_size, CM)
end

function MLJModelInterface.fit(model::ICNFModel, verbosity, X)
    x = collect(transpose(MLJModelInterface.matrix(X)))
    ps, st = LuxCore.setup(model.m.rng, model.m)
    ps = ComponentArray(ps)
    if model.m.resource isa CUDALibs
        gdev = gpu_device()
        x = gdev(x)
        ps = gdev(ps)
        st = gdev(st)
    end
    optfunc = OptimizationFunction(
        make_opt_loss(model.m, TrainMode(), st, model.loss),
        model.adtype,
    )
    optprob = OptimizationProblem(optfunc, ps)

    tst_overall = @timed for opt in model.optimizers
        tst_epochs = @timed for ep in 1:(model.n_epochs)
            if model.use_batch
                if model.compute_mode <: VectorMode
                    data = DataLoader(
                        (x,);
                        batchsize = -1,
                        shuffle = true,
                        partial = true,
                        parallel = false,
                        buffer = false,
                    )
                elseif model.compute_mode <: MatrixMode
                    data = DataLoader(
                        (x,);
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
                data = [(x,)]
            end
            optprob_re = remake(optprob; u0 = ps)
            tst_one = @timed res = solve(optprob_re, opt, data; progress = true)
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
    fitresult, cache, report
end

function MLJModelInterface.transform(model::ICNFModel, fitresult, Xnew)
    xnew = collect(transpose(MLJModelInterface.matrix(Xnew)))
    if model.m.resource isa CUDALibs
        gdev = gpu_device()
        xnew = gdev(xnew)
    end
    (ps, st) = fitresult

    tst = @timed if model.compute_mode <: VectorMode
        logp̂x = broadcast(x -> first(inference(model.m, TestMode(), x, ps, st)), eachcol(xnew))
    elseif model.compute_mode <: MatrixMode
        logp̂x = first(inference(model.m, TestMode(), xnew, ps, st))
    else
        error("Not Implemented")
    end

    @info(
        "Transforming",
        "elapsed time (seconds)" = tst.time,
        "garbage collection time (seconds)" = tst.gctime,
        "allocated (bytes)" = tst.bytes,
    )

    DataFrame(; px = exp.(logp̂x))
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
    input_scitype = Table{AbstractVector{ScientificTypes.Continuous}},
    target_scitype = Table{AbstractVector{ScientificTypes.Continuous}},
    output_scitype = Table{AbstractVector{ScientificTypes.Continuous}},
    supports_weights = false,
    load_path = "ContinuousNormalizingFlows.ICNFModel",
)

# Distributions interface

struct ICNFDist <: ICNFDistribution
    m::AbstractICNF
    mode::Mode
    ps::Any
    st::NamedTuple
end

function ICNFDist(mach::Machine{<:ICNFModel}, mode::Mode)
    (ps, st) = fitted_params(mach)
    ICNFDist(mach.model.m, mode, ps, st)
end

function Base.length(d::ICNFDist)
    d.m.nvars
end
function Base.eltype(d::ICNFDist)
    first(typeof(d.m).parameters)
end
function Distributions._logpdf(d::ICNFDist, x::AbstractVector{<:Real})
    if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        first(inference(d.m, d.mode, x, d.ps, d.st))
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        first(Distributions._logpdf(d, hcat(x)))
    else
        error("Not Implemented")
    end
end
function Distributions._logpdf(d::ICNFDist, A::AbstractMatrix{<:Real})
    if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        Distributions._logpdf.(d, eachcol(A))
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        first(inference(d.m, d.mode, A, d.ps, d.st))
    else
        error("Not Implemented")
    end
end
function Distributions._rand!(rng::AbstractRNG, d::ICNFDist, x::AbstractVector{<:Real})
    if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        x .= generate(d.m, d.mode, d.ps, d.st)
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        x .= Distributions._rand!(rng, d, hcat(x))
    else
        error("Not Implemented")
    end
end
function Distributions._rand!(rng::AbstractRNG, d::ICNFDist, A::AbstractMatrix{<:Real})
    if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        A .= hcat(Distributions._rand!.(rng, d, eachcol(A))...)
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        A .= generate(d.m, d.mode, d.ps, d.st, size(A, 2))
    else
        error("Not Implemented")
    end
end

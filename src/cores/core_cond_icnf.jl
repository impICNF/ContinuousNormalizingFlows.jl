export CondICNFModel, CondICNFDist

# MLJ interface

mutable struct CondICNFModel <: MLJICNF
    m::AbstractICNF
    loss::Function

    optimizers::Tuple
    n_epochs::Int
    adtype::ADTypes.AbstractADType

    use_batch::Bool
    batch_size::Int

    compute_mode::Type{<:ComputeMode}
end

function CondICNFModel(
    m::AbstractICNF{<:AbstractFloat, CM},
    loss::Function = loss;
    optimizers::Tuple = (Lion(),),
    n_epochs::Int = 300,
    adtype::ADTypes.AbstractADType = AutoZygote(),
    use_batch::Bool = true,
    batch_size::Int = 32,
) where {CM <: ComputeMode}
    CondICNFModel(m, loss, optimizers, n_epochs, adtype, use_batch, batch_size, CM)
end

function MLJModelInterface.fit(model::CondICNFModel, verbosity, XY)
    X, Y = XY
    x = collect(transpose(MLJModelInterface.matrix(X)))
    y = collect(transpose(MLJModelInterface.matrix(Y)))
    ps, st = LuxCore.setup(model.m.rng, model.m)
    ps = ComponentArray(ps)
    if model.m.resource isa CUDALibs
        gdev = gpu_device()
        x = gdev(x)
        y = gdev(y)
        ps = gdev(ps)
        st = gdev(st)
    end
    optfunc = OptimizationFunction(
        let mm = model.m, md = TrainMode(), st = st
            (u, p, xs_, ys_) -> model.loss(mm, md, xs_, ys_, u, st)
        end,
        model.adtype,
    )
    optprob = OptimizationProblem(optfunc, ps)
    tst_overall = @timed for opt in model.optimizers
        tst_epochs = @timed for ep in 1:(model.n_epochs)
            if model.use_batch
                if model.compute_mode <: VectorMode
                    data = DataLoader(
                        (x, y);
                        batchsize = -1,
                        shuffle = true,
                        partial = true,
                        parallel = false,
                        buffer = false,
                    )
                elseif model.compute_mode <: MatrixMode
                    data = DataLoader(
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
            optprob_re = remake(optprob; u0 = ps)
            tst_one = @timed res = solve(optprob_re, opt, data; progress = true)
            ps .= res.u
            @info(
                "Fitting (epoch: $ep of $(model.n_epochs)) - $(typeof(opt).name.name)",
                "elapsed time (seconds)" = tst_one.time,
                "garbage collection time (seconds)" = tst_one.gctime,
            )
        end
        @info(
            "Fitting (all epochs) - $(typeof(opt).name.name)",
            "elapsed time (seconds)" = tst_epochs.time,
            "garbage collection time (seconds)" = tst_epochs.gctime,
        )
    end
    @info(
        "Fitting - Overall",
        "elapsed time (seconds)" = tst_overall.time,
        "garbage collection time (seconds)" = tst_overall.gctime,
    )

    fitresult = (ps, st)
    cache = nothing
    report = (stats = tst_overall,)
    fitresult, cache, report
end

function MLJModelInterface.transform(model::CondICNFModel, fitresult, XYnew)
    Xnew, Ynew = XYnew
    xnew = collect(transpose(MLJModelInterface.matrix(Xnew)))
    ynew = collect(transpose(MLJModelInterface.matrix(Ynew)))
    if model.m.resource isa CUDALibs
        gdev = gpu_device()
        xnew = gdev(xnew)
        ynew = gdev(ynew)
    end
    (ps, st) = fitresult

    if model.compute_mode <: VectorMode
        tst = @timed logp̂x = broadcast(
            let mm = model.m, md = TestMode(), ps = ps, st = st
                (x, y) -> first(inference(mm, md, x, y, ps, st))
            end,
            eachcol(xnew),
            eachcol(ynew),
        )
    elseif model.compute_mode <: MatrixMode
        tst = @timed logp̂x = first(inference(model.m, TestMode(), xnew, ynew, ps, st))
    else
        error("Not Implemented")
    end
    @info(
        "Transforming",
        "elapsed time (seconds)" = tst.time,
        "garbage collection time (seconds)" = tst.gctime,
    )

    DataFrame(; px = exp.(logp̂x))
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
        Table{AbstractVector{ScientificTypes.Continuous}},
        Table{AbstractVector{ScientificTypes.Continuous}},
    },
    target_scitype = Table{AbstractVector{ScientificTypes.Continuous}},
    output_scitype = Table{AbstractVector{ScientificTypes.Continuous}},
    supports_weights = false,
    load_path = "ContinuousNormalizingFlows.CondICNFModel",
)

# Distributions interface

struct CondICNFDist <: ICNFDistribution
    m::AbstractICNF
    mode::Mode
    ys::AbstractVecOrMat{<:Real}
    ps::Any
    st::NamedTuple
end

function CondICNFDist(
    mach::Machine{<:CondICNFModel},
    mode::Mode,
    ys::AbstractVecOrMat{<:Real},
)
    (ps, st) = fitted_params(mach)
    CondICNFDist(mach.model.m, mode, ys, ps, st)
end

function Base.length(d::CondICNFDist)
    d.m.nvars
end
function Base.eltype(d::CondICNFDist)
    first(typeof(d.m).parameters)
end
function Distributions._logpdf(d::CondICNFDist, x::AbstractVector{<:Real})
    if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        first(inference(d.m, d.mode, x, d.ys, d.ps, d.st))
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        first(Distributions._logpdf(d, hcat(x)))
    else
        error("Not Implemented")
    end
end
function Distributions._logpdf(d::CondICNFDist, A::AbstractMatrix{<:Real})
    if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        broadcast(let d = d
            x -> Distributions._logpdf(d, x)
        end, eachcol(A))
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        first(inference(d.m, d.mode, A, d.ys[:, begin:size(A, 2)], d.ps, d.st))
    else
        error("Not Implemented")
    end
end
function Distributions._rand!(rng::AbstractRNG, d::CondICNFDist, x::AbstractVector{<:Real})
    if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        x .= generate(d.m, d.mode, d.ys, d.ps, d.st)
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        x .= Distributions._rand!(rng, d, hcat(x))
    else
        error("Not Implemented")
    end
end
function Distributions._rand!(rng::AbstractRNG, d::CondICNFDist, A::AbstractMatrix{<:Real})
    if d.m isa AbstractICNF{<:AbstractFloat, <:VectorMode}
        A .= hcat(broadcast(let rng = rng, d = d
            x -> Distributions._rand!(rng, d, x)
        end, eachcol(A))...)
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:MatrixMode}
        A .= generate(d.m, d.mode, d.ys[:, begin:size(A, 2)], d.ps, d.st, size(A, 2))
    else
        error("Not Implemented")
    end
end

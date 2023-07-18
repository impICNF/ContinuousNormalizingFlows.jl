export ICNFModel, ICNFDist

# SciML interface

function loss_f(icnf::AbstractICNF, loss::Function, st::Any)::Function
    function f(ps, θ, xs)
        loss(icnf, TrainMode(), xs, ps, st)
    end
    f
end

# MLJ interface

mutable struct ICNFModel <: MLJICNF
    m::AbstractICNF
    loss::Function

    optimizers::AbstractVector
    n_epochs::Integer
    adtype::ADTypes.AbstractADType

    use_batch::Bool
    batch_size::Integer
    have_callback::Bool

    resource::AbstractResource
    data_type::Type{<:AbstractFloat}
    array_type::Type{<:AbstractArray}
    compute_mode::Type{<:ComputeMode}
end

function ICNFModel(
    m::AbstractICNF{T, AT, CM},
    loss::Function = loss;
    optimizers::AbstractVector = [Optimisers.Adam()],
    n_epochs::Integer = 300,
    adtype::ADTypes.AbstractADType = AutoZygote(),
    use_batch::Bool = true,
    batch_size::Integer = 32,
    have_callback::Bool = true,
    resource::AbstractResource = CPU1(),
) where {T <: AbstractFloat, AT <: AbstractArray, CM <: ComputeMode}
    ICNFModel(
        m,
        loss,
        optimizers,
        n_epochs,
        adtype,
        use_batch,
        batch_size,
        have_callback,
        resource,
        T,
        AT,
        CM,
    )
end

function MLJModelInterface.fit(model::ICNFModel, verbosity, X)
    rng = Random.default_rng()
    x = collect(transpose(MLJModelInterface.matrix(X)))
    ps, st = LuxCore.setup(rng, model.m)
    if !(model.m isa FluxCompatLayer)
        ps = ComponentArray(ps)
    end
    if model.resource isa CUDALibs
        gdev = gpu_device()
        x = gdev(x)
        ps = gdev(ps)
        st = gdev(st)
    else
        x = convert(model.array_type{model.data_type}, x)
        if model.m isa FluxCompatLayer
            ps = convert(model.array_type{model.data_type}, ps)
        else
            ps = ComponentArray{model.data_type}(ps)
        end
    end
    _loss = loss_f(model.m, model.loss, st)
    optfunc = OptimizationFunction(_loss, model.adtype)
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
            if model.have_callback
                prgr = Progress(
                    length(data);
                    desc = "Fitting (epoch: $ep of $(model.n_epochs)): ",
                    showspeed = true,
                )
                _callback = callback_f(model.m, prgr)
                tst_one =
                    @timed res = solve(optprob_re, opt, data; callback = _callback)
                ProgressMeter.finish!(prgr)
            else
                tst_one = @timed res = solve(optprob_re, opt, data)
            end
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

function MLJModelInterface.transform(model::ICNFModel, fitresult, Xnew)
    xnew = collect(transpose(MLJModelInterface.matrix(Xnew)))
    if model.resource isa CUDALibs
        gdev = gpu_device()
        xnew = gdev(xnew)
    end
    (ps, st) = fitresult

    if model.compute_mode <: VectorMode
        tst = @timed logp̂x = broadcast(
            x -> first(inference(model.m, TestMode(), x, ps, st)),
            eachcol(xnew),
        )
    elseif model.compute_mode <: MatrixMode
        tst = @timed logp̂x = first(inference(model.m, TestMode(), xnew, ps, st))
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
    st::Any
end

function ICNFDist(mach::Machine{<:ICNFModel}, mode::Mode)
    (ps, st) = fitted_params(mach)
    ICNFDist(mach.model.m, mode, ps, st)
end

Base.length(d::ICNFDist) = d.m.nvars
Base.eltype(d::ICNFDist) = typeof(d.m).parameters[1]
function Distributions._logpdf(d::ICNFDist, x::AbstractVector{<:Real})
    if d.m isa AbstractICNF{<:AbstractFloat, <:AbstractArray, <:VectorMode}
        first(inference(d.m, d.mode, x, d.ps, d.st))
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:AbstractArray, <:MatrixMode}
        first(Distributions._logpdf(d, hcat(x)))
    else
        error("Not Implemented")
    end
end
function Distributions._logpdf(d::ICNFDist, A::AbstractMatrix{<:Real})
    if d.m isa AbstractICNF{<:AbstractFloat, <:AbstractArray, <:VectorMode}
        broadcast(x -> Distributions._logpdf(d, x), eachcol(A))
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:AbstractArray, <:MatrixMode}
        first(inference(d.m, d.mode, A, d.ps, d.st))
    else
        error("Not Implemented")
    end
end
function Distributions._rand!(rng::AbstractRNG, d::ICNFDist, x::AbstractVector{<:Real})
    if d.m isa AbstractICNF{<:AbstractFloat, <:AbstractArray, <:VectorMode}
        x .= generate(d.m, d.mode, d.ps, d.st; rng)
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:AbstractArray, <:MatrixMode}
        x .= Distributions._rand!(rng, d, hcat(x))
    else
        error("Not Implemented")
    end
end
function Distributions._rand!(rng::AbstractRNG, d::ICNFDist, A::AbstractMatrix{<:Real})
    if d.m isa AbstractICNF{<:AbstractFloat, <:AbstractArray, <:VectorMode}
        A .= hcat(broadcast(x -> Distributions._rand!(rng, d, x), eachcol(A))...)
    elseif d.m isa AbstractICNF{<:AbstractFloat, <:AbstractArray, <:MatrixMode}
        A .= generate(d.m, d.mode, d.ps, d.st, size(A, 2); rng)
    else
        error("Not Implemented")
    end
end

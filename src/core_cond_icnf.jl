export CondICNFModel, CondICNFDist

# SciML interface

function loss_f(icnf::AbstractCondICNF, loss::Function, st::Any)::Function
    function f(ps, θ, xs, ys)
        loss(icnf, xs, ys, ps, st)
    end
    f
end

# MLJ interface

mutable struct CondICNFModel <: MLJICNF
    m::AbstractCondICNF
    loss::Function

    optimizers::AbstractVector
    n_epochs::Integer
    adtype::SciMLBase.AbstractADType

    use_batch::Bool
    batch_size::Integer
    have_callback::Bool

    resource::AbstractResource
    data_type::Type{<:AbstractFloat}
    array_type::Type{<:AbstractArray}
    compute_mode::Type{<:ComputeMode}
end

function CondICNFModel(
    m::AbstractCondICNF{T, AT, CM},
    loss::Function = loss,
    ;
    optimizers::AbstractVector = [Optimisers.Adam()],
    n_epochs::Integer = 128,
    adtype::SciMLBase.AbstractADType = Optimization.AutoZygote(),
    use_batch::Bool = true,
    batch_size::Integer = 128,
    have_callback::Bool = true,
    resource::AbstractResource = CPU1(),
) where {T <: AbstractFloat, AT <: AbstractArray, CM <: ComputeMode}
    CondICNFModel(
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

function MLJModelInterface.fit(model::CondICNFModel, verbosity, XY)
    rng = Random.default_rng()
    X, Y = XY
    x = collect(transpose(MLJModelInterface.matrix(X)))
    y = collect(transpose(MLJModelInterface.matrix(Y)))
    ps, st = LuxCore.setup(rng, model.m)
    if !(model.m isa FluxCompatLayer)
        ps = ComponentArray(ps)
    end
    if model.resource isa CUDALibs
        x = Lux.gpu(x)
        y = Lux.gpu(y)
        ps = Lux.gpu(ps)
        st = Lux.gpu(st)
    else
        x = convert(model.array_type{model.data_type}, x)
        y = convert(model.array_type{model.data_type}, y)
        if model.m isa FluxCompatLayer
            ps = convert(model.array_type{model.data_type}, ps)
        else
            ps = ComponentArray{model.data_type}(ps)
        end
    end
    if model.use_batch
        if model.compute_mode <: VectorMode
            data = DataLoader(
                (x, y);
                batchsize = -1,
                partial = true,
                shuffle = true,
                parallel = true,
                buffer = false,
            )
        elseif model.compute_mode <: MatrixMode
            data = DataLoader(
                (x, y);
                batchsize = model.batch_size,
                partial = true,
                shuffle = true,
                parallel = true,
                buffer = false,
            )
        else
            error("Not Implemented")
        end
    else
        data = [(x, y)]
    end
    ncdata = ncycle(data, model.n_epochs)
    _loss = loss_f(model.m, model.loss, st)
    optfunc = OptimizationFunction(_loss, model.adtype)
    optprob = OptimizationProblem(optfunc, ps)
    tst = @timed for opt in model.optimizers
        optprob_re = remake(optprob; u0 = ps)
        if model.have_callback
            prgr = Progress(length(ncdata); desc = "Training: ", showspeed = true)
            _callback = callback_f(model.m, prgr)
            tst_one = @timed res = solve(optprob_re, opt, ncdata; callback = _callback)
            ProgressMeter.finish!(prgr)
            @info(
                "Fitting - $(typeof(opt).name.name)",
                "elapsed time (seconds)" = tst_one.time,
                "garbage collection time (seconds)" = tst_one.gctime,
            )
        else
            res = solve(optprob_re, opt, ncdata)
        end
        ps .= res.u
    end
    @info(
        "Fitting - Overall",
        "elapsed time (seconds)" = tst.time,
        "garbage collection time (seconds)" = tst.gctime,
    )

    fitresult = (ps, st)
    cache = nothing
    report = (stats = tst,)
    fitresult, cache, report
end

function MLJModelInterface.transform(model::CondICNFModel, fitresult, XYnew)
    Xnew, Ynew = XYnew
    xnew = collect(transpose(MLJModelInterface.matrix(Xnew)))
    ynew = collect(transpose(MLJModelInterface.matrix(Ynew)))
    if model.resource isa CUDALibs
        xnew = Lux.gpu(xnew)
        ynew = Lux.gpu(ynew)
    end
    (ps, st) = fitresult

    if model.compute_mode <: VectorMode
        tst = @timed logp̂x = broadcast(
            ((x, y),) -> first(inference(model.m, TestMode(), x, y, ps, st)),
            zip(eachcol(xnew), eachcol(ynew)),
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
    package_name = "ICNF",
    package_uuid = "9bd0f7d2-bd29-441d-bcde-0d11364d2762",
    package_url = "https://github.com/impICNF/ICNF.jl",
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
    load_path = "ICNF.CondICNFModel",
)

# Distributions interface

struct CondICNFDist <: ICNFDistribution
    m::AbstractCondICNF
    ys::AbstractVecOrMat{<:Real}
    ps::Any
    st::Any
end

Base.length(d::CondICNFDist) = d.m.nvars
Base.eltype(d::CondICNFDist) = typeof(d.m).parameters[1]
function Distributions._logpdf(d::CondICNFDist, x::AbstractVector{<:Real})
    if d.m isa AbstractCondICNF{<:AbstractFloat, <:AbstractArray, <:VectorMode}
        first(inference(d.m, TestMode(), x, d.ys, d.ps, d.st))
    elseif d.m isa AbstractCondICNF{<:AbstractFloat, <:AbstractArray, <:MatrixMode}
        first(Distributions._logpdf(d, hcat(x)))
    else
        error("Not Implemented")
    end
end
function Distributions._logpdf(d::CondICNFDist, A::AbstractMatrix{<:Real})
    if d.m isa AbstractCondICNF{<:AbstractFloat, <:AbstractArray, <:VectorMode}
        broadcast(x -> Distributions._logpdf(d, x), eachcol(A))
    elseif d.m isa AbstractCondICNF{<:AbstractFloat, <:AbstractArray, <:MatrixMode}
        first(inference(d.m, TestMode(), A, d.ys[:, 1:size(A, 2)], d.ps, d.st))
    else
        error("Not Implemented")
    end
end
function Distributions._rand!(rng::AbstractRNG, d::CondICNFDist, x::AbstractVector{<:Real})
    if d.m isa AbstractCondICNF{<:AbstractFloat, <:AbstractArray, <:VectorMode}
        x .= generate(d.m, TestMode(), d.ys, d.ps, d.st; rng)
    elseif d.m isa AbstractCondICNF{<:AbstractFloat, <:AbstractArray, <:MatrixMode}
        x .= Distributions._rand!(rng, d, hcat(x))
    else
        error("Not Implemented")
    end
end
function Distributions._rand!(rng::AbstractRNG, d::CondICNFDist, A::AbstractMatrix{<:Real})
    if d.m isa AbstractCondICNF{<:AbstractFloat, <:AbstractArray, <:VectorMode}
        A .= hcat(broadcast(x -> Distributions._rand!(rng, d, x), eachcol(A))...)
    elseif d.m isa AbstractCondICNF{<:AbstractFloat, <:AbstractArray, <:MatrixMode}
        A .= generate(d.m, TestMode(), d.ys[:, 1:size(A, 2)], d.ps, d.st, size(A, 2); rng)
    else
        error("Not Implemented")
    end
end

export ICNFModel, ICNFDist

# SciML interface

function loss_f(icnf::AbstractICNF, loss::Function, st::Any)::Function
    function f(ps, θ, xs)
        loss(icnf, xs, ps, st)
    end
    f
end

# MLJ interface

mutable struct ICNFModel <: MLJICNF
    m::AbstractICNF
    loss::Function

    optimizer::Any
    n_epochs::Integer
    adtype::SciMLBase.AbstractADType

    use_batch::Bool
    batch_size::Integer

    resource::AbstractResource
    data_type::Type{<:AbstractFloat}
    array_type::Type{<:AbstractArray}
    compute_mode::Type{<:ComputeMode}
end

function ICNFModel(
    m::AbstractICNF{T, AT, CM},
    loss::Function = loss,
    ;
    optimizer::Any = Optimisers.Adam(),
    n_epochs::Integer = 128,
    adtype::SciMLBase.AbstractADType = Optimization.AutoZygote(),
    use_batch::Bool=true,
    batch_size::Integer = 128,
    resource::AbstractResource = CPU1(),
) where {T <: AbstractFloat, AT <: AbstractArray, CM <: ComputeMode}
    ICNFModel(m, loss, optimizer, n_epochs, adtype, use_batch, batch_size, resource, T, AT, CM)
end

function MLJModelInterface.fit(model::ICNFModel, verbosity, X)
    rng = Random.default_rng()
    x = collect(transpose(MLJModelInterface.matrix(X)))
    ps, st = LuxCore.setup(rng, model.m)
    if !(model.m isa MyFluxLayer)
        ps = ComponentArray(ps)
    end
    if model.resource isa CUDALibs
        x = Lux.gpu(x)
        ps = Lux.gpu(ps)
        st = Lux.gpu(st)
    else
        x = convert(model.array_type{model.data_type}, x)
        if model.m isa MyFluxLayer
            ps = convert(model.array_type{model.data_type}, ps)
        else
            ps = ComponentArray{model.data_type}(ps)
        end
    end
    if model.use_batch
        if model.compute_mode <: VectorMode
            data = DataLoader(
                (x,);
                batchsize = -1,
                partial = true,
                shuffle = true,
                parallel = true,
                buffer = true,
            )
        elseif model.compute_mode <: MatrixMode
            data = DataLoader(
                (x,);
                batchsize = model.batch_size,
                partial = true,
                shuffle = true,
                parallel = true,
                buffer = true,
            )
        else
            error("Not Implemented")
        end
    else
        data = [(x,)]
    end
    ncdata = ncycle(data, model.n_epochs)
    _loss = loss_f(model.m, model.loss, st)
    _callback = callback_f(model.m)
    optfunc = OptimizationFunction(_loss, model.adtype)
    optprob = OptimizationProblem(optfunc, ps)
    tst = @timed res = solve(optprob, model.optimizer, ncdata; callback = _callback)
    ps .= res.u
    @info(
        "Fitting",
        "elapsed time (seconds)" = tst.time,
        "garbage collection time (seconds)" = tst.gctime,
    )

    fitresult = (ps, st)
    cache = nothing
    report = (stats = tst,)
    fitresult, cache, report
end

function MLJModelInterface.transform(model::ICNFModel, fitresult, Xnew)
    xnew = collect(transpose(MLJModelInterface.matrix(Xnew)))
    if model.resource isa CUDALibs
        xnew = Lux.gpu(xnew)
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
    package_name = "ICNF",
    package_uuid = "9bd0f7d2-bd29-441d-bcde-0d11364d2762",
    package_url = "https://github.com/impICNF/ICNF.jl",
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
    load_path = "ICNF.ICNFModel",
)

# Distributions interface

struct ICNFDist <: ICNFDistribution
    m::AbstractICNF
    ps::Any
    st::Any
end

Base.length(d::ICNFDist) = d.m.nvars
Base.eltype(d::ICNFDist) = typeof(d.m).parameters[1]
function Distributions._logpdf(d::ICNFDist, x::AbstractVector{<:Real})
    if d.m isa AbstractICNF{<:AbstractFloat, <:AbstractArray, <:VectorMode}
        first(inference(d.m, TestMode(), x, d.ps, d.st))
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
        first(inference(d.m, TestMode(), A, d.ps, d.st))
    else
        error("Not Implemented")
    end
end
function Distributions._rand!(rng::AbstractRNG, d::ICNFDist, x::AbstractVector{<:Real})
    if d.m isa AbstractICNF{<:AbstractFloat, <:AbstractArray, <:VectorMode}
        x .= generate(d.m, TestMode(), d.ps, d.st; rng)
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
        A .= generate(d.m, TestMode(), d.ps, d.st, size(A, 2); rng)
    else
        error("Not Implemented")
    end
end

export loss_f, callback_f, ICNFModel, ICNFDist

# -- SciML interface

function loss_f(
    icnf::AbstractICNF{T, AT},
    loss::Function,
    st::Any,
)::Function where {T <: AbstractFloat, AT <: AbstractArray}
    function f(ps, θ, xs)
        loss(icnf, xs, ps, st)
    end
    f
end

function callback_f(
    icnf::AbstractICNF{T, AT},
    loss::Function,
    data::DataLoader{T3},
    st::Any,
)::Function where {
    T <: AbstractFloat,
    AT <: AbstractArray,
    T2 <: AbstractMatrix{<:Real},
    T3 <: Tuple{T2},
}
    xs, = first(data)
    function f(ps, l)
        vl = loss(icnf, xs, ps, st)
        @info "Training" loss = vl
        false
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

    batch_size::Integer
    resource::AbstractResource
    data_type::Type{<:AbstractFloat}
    array_type::Type{<:AbstractArray}
end

function ICNFModel(
    m::AbstractICNF{T, AT},
    loss::Function = loss,
    ;
    optimizer::Any = default_optimizer,
    n_epochs::Integer = 128,
    adtype::SciMLBase.AbstractADType = Optimization.AutoZygote(),
    batch_size::Integer = 128,
    resource::AbstractResource = CPU1(),
) where {T <: AbstractFloat, AT <: AbstractArray}
    ICNFModel(m, loss, optimizer, n_epochs, adtype, batch_size, resource, T, AT)
end

function MLJModelInterface.fit(model::ICNFModel, verbosity, X)
    rng = Random.default_rng()
    x = collect(transpose(MLJModelInterface.matrix(X)))
    ps, st = LuxCore.setup(rng, model.m)
    ps = ComponentArray(ps)
    if model.resource isa CUDALibs
        x = gpu(x)
        ps = gpu(ps)
        st = gpu(st)
    else
        x = model.array_type{model.data_type}(x)
        ps = ComponentArray{model.data_type}(ps)
    end
    data = DataLoader((x,); batchsize = model.batch_size, shuffle = true, partial = true)
    ncdata = ncycle(data, model.n_epochs)
    initial_loss_value = model.loss(model.m, first(data)..., ps, st)
    _loss = loss_f(model.m, model.loss, st)
    _callback = callback_f(model.m, model.loss, data, st)
    optfunc = OptimizationFunction(_loss, model.adtype)
    optprob = OptimizationProblem(optfunc, ps)
    tst = @timed res = solve(optprob, model.optimizer, ncdata; callback = _callback)
    ps = res.u
    final_loss_value = model.loss(model.m, first(data)..., ps, st)
    @info(
        "Fitting",
        "elapsed time (seconds)" = tst.time,
        "garbage collection time (seconds)" = tst.gctime,
    )

    fitresult = (ps, st)
    cache = nothing
    report = (
        stats = tst,
        initial_loss_value = initial_loss_value,
        final_loss_value = final_loss_value,
    )
    fitresult, cache, report
end

function MLJModelInterface.transform(model::ICNFModel, fitresult, Xnew)
    xnew = collect(transpose(MLJModelInterface.matrix(Xnew)))
    if model.resource isa CUDALibs
        xnew = gpu(xnew)
    end
    (ps, st) = fitresult

    tst = @timed logp̂x =
        broadcast(x -> first(inference(model.m, TestMode(), x, ps, st)), eachcol(xnew))
    @info(
        "Transforming",
        "elapsed time (seconds)" = tst.time,
        "garbage collection time (seconds)" = tst.gctime,
    )

    DataFrame(; px = exp.(logp̂x))
end

function MLJModelInterface.fitted_params(model::ICNFModel, fitresult)
    (ps, st) = fitresult
    (learned_parameters = ps, states = st)
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
    first(inference(d.m, TestMode(), x, d.ps, d.st))
end
function Distributions._logpdf(d::ICNFDist, A::AbstractMatrix{<:Real})
    broadcast(x -> Distributions._logpdf(d, x), eachcol(A))
end
function Distributions._rand!(rng::AbstractRNG, d::ICNFDist, x::AbstractVector{<:Real})
    x .= generate(d.m, TestMode(), d.ps, d.st; rng)
end
function Distributions._rand!(rng::AbstractRNG, d::ICNFDist, A::AbstractMatrix{<:Real})
    A .= hcat(broadcast(x -> Distributions._rand!(rng, d, x), eachcol(A))...)
end

abstract type MLJICNF{AICNF <: AbstractICNF} <: MLJModelInterface.Unsupervised end

function MLJModelInterface.fitted_params(::MLJICNF, fitresult)
    (ps, st) = fitresult
    return (learned_parameters = ps, states = st)
end

function make_opt_loss(icnf::AbstractICNF, mode::Mode, st::NamedTuple, loss_::Function)
    function opt_loss(u::Any, (xs,)::Tuple{<:Any})
        return loss_(icnf, mode, xs, u, st)
    end

    function opt_loss(u::Any, (xs, ys)::Tuple{<:Any, <:Any})
        return loss_(icnf, mode, xs, ys, u, st)
    end

    return opt_loss
end

function make_dataloader(::AbstractICNF{<:AbstractFloat, <:VectorMode}, ::Int, data::Tuple)
    return MLUtils.DataLoader(data; batchsize = -1, shuffle = true, partial = true)
end

function make_dataloader(
    ::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    batchsize::Int,
    data::Tuple,
)
    return MLUtils.DataLoader(
        data;
        batchsize = get_batchsize(Val(iszero(batchsize)), batchsize, data),
        shuffle = true,
        partial = true,
    )
end

function get_batchsize(::Val{true}, ::Int, data::Tuple)
    return last(maximum(size.(data)))
end

function get_batchsize(::Val{false}, batchsize::Int, ::Tuple)
    return batchsize
end

function get_logp̂x(
    icnf::AbstractICNF{T, <:VectorMode, INPLACE, false},
    xnew::Any,
    ps::Any,
    st::NamedTuple,
) where {T <: AbstractFloat, INPLACE}
    @warn "to compute by vectors, data should be a vector." maxlog = 1
    return broadcast(
        function (x::AbstractVector{<:Real})
            return first(inference(icnf, TestMode(), x, ps, st))
        end,
        collect(collect.(eachcol(xnew))),
    )
end

function get_logp̂x(
    icnf::AbstractICNF{T, <:MatrixMode, INPLACE, false},
    xnew::Any,
    ps::Any,
    st::NamedTuple,
) where {T <: AbstractFloat, INPLACE}
    return first(inference(icnf, TestMode(), xnew, ps, st))
end

function get_logp̂x(
    icnf::AbstractICNF{T, <:VectorMode, INPLACE, true},
    xnew::Any,
    ynew::Any,
    ps::Any,
    st::NamedTuple,
) where {T <: AbstractFloat, INPLACE}
    @warn "to compute by vectors, data should be a vector." maxlog = 1
    broadcast(
        function (x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
            return first(inference(icnf, TestMode(), x, y, ps, st))
        end,
        collect(collect.(eachcol(xnew))),
        collect(collect.(eachcol(ynew))),
    )
end

function get_logp̂x(
    icnf::AbstractICNF{T, <:MatrixMode, INPLACE, true},
    xnew::Any,
    ynew::Any,
    ps::Any,
    st::NamedTuple,
) where {T <: AbstractFloat, INPLACE}
    return first(inference(icnf, TestMode(), xnew, ynew, ps, st))
end

function make_opt_callback(n::Int)
    function opt_callback(state::Any, l::Any)
        if isone(state.iter % n)
            println("Iteration: $(state.iter) | Loss: $l")
        end
        return false
    end

    return opt_callback
end

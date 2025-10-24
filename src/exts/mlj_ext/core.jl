function MLJModelInterface.fitted_params(::MLJICNF, fitresult)
    (ps, st) = fitresult
    return (learned_parameters = ps, states = st)
end

function make_opt_loss(icnf::AbstractICNF, mode::Mode, st::NamedTuple, loss_::Function)
    function opt_loss(u::Any, data::Tuple{<:Any})
        xs, = data
        return loss_(icnf, mode, xs, u, st)
    end

    function opt_loss(u::Any, data::Tuple{<:Any, <:Any})
        xs, ys = data
        return loss_(icnf, mode, xs, ys, u, st)
    end

    return opt_loss
end

function make_dataloader(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    ::Int,
    data::Tuple,
)
    return MLUtils.DataLoader(data; batchsize = -1, shuffle = true, partial = true)
end

function make_dataloader(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    batchsize::Int,
    data::Tuple,
)
    return MLUtils.DataLoader(
        data;
        batchsize = if iszero(batchsize)
            last(maximum(size.(data)))
        else
            batchsize
        end,
        shuffle = true,
        partial = true,
    )
end

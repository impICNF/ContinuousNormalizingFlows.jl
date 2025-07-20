function MLJModelInterface.fitted_params(::MLJICNF, fitresult)
    (ps, st) = fitresult
    return (learned_parameters = ps, states = st)
end

function make_opt_loss(
    icnf::AbstractICNF{T, CM, INPLACE, COND},
    mode::Mode,
    st::NamedTuple,
    loss_::Function,
) where {T, CM, INPLACE, COND}
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

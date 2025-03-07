function MLJModelInterface.fitted_params(::MLJICNF, fitresult)
    (ps, st) = fitresult
    return (learned_parameters = ps, states = st)
end

@inline function make_opt_loss(
    icnf::AbstractICNF{T, CM, INPLACE, COND},
    mode::Mode,
    st::NamedTuple,
    loss_::Function,
) where {T, CM, INPLACE, COND}
    function opt_loss_org(u, data)
        xs, = data
        return loss_(icnf, mode, xs, u, st)
    end

    function opt_loss_cond(u, data)
        xs, ys = data
        return loss_(icnf, mode, xs, ys, u, st)
    end

    return ifelse(COND, opt_loss_cond, opt_loss_org)
end

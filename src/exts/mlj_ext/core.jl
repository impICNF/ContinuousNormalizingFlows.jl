function MLJModelInterface.fitted_params(::MLJICNF, fitresult)
    (ps, st) = fitresult
    (learned_parameters = ps, states = st)
end

@inline function make_opt_loss(
    icnf::AbstractICNF{T, CM, INPLACE, COND},
    mode::Mode,
    st::NamedTuple,
    loss_::Function,
) where {T, CM, INPLACE, COND}
    function opt_loss_org(u, p, data)
        xs, = data
        loss_(icnf, mode, xs, u, st)
    end

    function opt_loss_cond(u, p, data)
        xs, ys = data
        loss_(icnf, mode, xs, ys, u, st)
    end

    ifelse(COND, opt_loss_cond, opt_loss_org)
end

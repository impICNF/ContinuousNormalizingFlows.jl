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
    function opt_loss_org(u, p, xs)
        loss_(icnf, mode, xs, u, st)
    end

    function opt_loss_cond(u, p, xs, ys)
        loss_(icnf, mode, xs, ys, u, st)
    end

    ifelse(COND, opt_loss_cond, opt_loss_org)
end

function Base.length(d::ICNFDistribution)
    d.m.nvars
end

function Base.eltype(::ICNFDistribution{AICNF}) where {AICNF <: AbstractICNF}
    first(AICNF.parameters)
end

function Base.broadcastable(d::ICNFDistribution)
    Ref(d)
end

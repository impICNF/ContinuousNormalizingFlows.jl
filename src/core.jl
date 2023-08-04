# SciML interface

function callback_f(ps, l, icnf::AbstractFlows, prgr::Progress, itr_n::AbstractVector)
    ProgressMeter.next!(
        prgr;
        showvalues = [(:loss_value, l), (:iteration, itr_n), (:last_update, Dates.now())],
    )
    itr_n[] += one(itr_n[])
    false
end

# MLJ interface

function MLJModelInterface.fitted_params(model::MLJICNF, fitresult)
    (ps, st) = fitresult
    (learned_parameters = ps, states = st)
end

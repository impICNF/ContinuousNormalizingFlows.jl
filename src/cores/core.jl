# SciML interface

function callback_f(ps, l, ::AbstractICNF, prgr::Progress, itr_n::AbstractArray)
    ProgressMeter.next!(
        prgr;
        showvalues = [
            (:loss_value, l),
            (:iteration, only(itr_n)),
            (:last_update, Dates.now()),
        ],
    )
    itr_n[] += one(only(itr_n))
    false
end

# MLJ interface

function MLJModelInterface.fitted_params(::MLJICNF, fitresult)
    (ps, st) = fitresult
    (learned_parameters = ps, states = st)
end

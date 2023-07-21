# SciML interface

function callback_f(icnf::AbstractFlows, prgr::Progress)
    itr_n = 1
    function f(ps, l)
        ProgressMeter.next!(
            prgr;
            showvalues = [
                (:loss_value, l),
                (:iteration, itr_n),
                (:last_update, Dates.now()),
            ],
        )
        itr_n += one(itr_n)
        false
    end
    f
end

# MLJ interface

function MLJModelInterface.fitted_params(model::MLJICNF, fitresult)
    (ps, st) = fitresult
    (learned_parameters = ps, states = st)
end

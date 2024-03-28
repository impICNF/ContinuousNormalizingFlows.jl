# MLJ interface

function MLJModelInterface.fitted_params(::MLJICNF, fitresult)
    (ps, st) = fitresult
    (learned_parameters = ps, states = st)
end

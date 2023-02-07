# SciML interface

function callback_f(icnf::AbstractFlows)::Function
    function f(ps, l)
        @info "Training" loss = l
        false
    end
    f
end

# MLJ interface

function MLJModelInterface.fitted_params(model::MLJICNF, fitresult)
    (ps, st) = fitresult
    (learned_parameters = ps, states = st)
end

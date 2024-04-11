@testset "Instability" begin
    JET.test_package(
        ContinuousNormalizingFlows;
        target_modules = [ContinuousNormalizingFlows],
        mode = :sound,
    )

    nvars = 2^3
    naugs = nvars
    n_in = nvars + naugs
    n = 2^6
    nn = Lux.Dense(n_in => n_in, tanh)

    icnf = construct(
        RNODE,
        nn,
        nvars,
        naugs;
        compute_mode = DIVecJacMatrixMode,
        tspan = (0.0f0, 13.0f0),
        steer_rate = 1.0f-1,
        λ₃ = 1.0f-2,
    )
    ps, st = Lux.setup(icnf.rng, icnf)
    ps = ComponentArrays.ComponentArray(ps)
    r = rand(icnf.rng, Float32, nvars, n)

    loss(icnf, TrainMode(), r, ps, st)
    JET.test_call(
        loss,
        Base.typesof(icnf, TrainMode(), r, ps, st);
        target_modules = [ContinuousNormalizingFlows],
        mode = :sound,
    )
    JET.test_opt(
        loss,
        Base.typesof(icnf, TrainMode(), r, ps, st);
        target_modules = [ContinuousNormalizingFlows],
    )
end

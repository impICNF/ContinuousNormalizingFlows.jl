Test.@testset "CheckByJET" begin
    JET.test_package(
        ContinuousNormalizingFlows;
        target_modules = [ContinuousNormalizingFlows],
        mode = :typo,
    )

    nvars = 2^3
    naugs = nvars
    n_in = nvars + naugs
    n = 2^6
    nn = Lux.Chain(Lux.Dense(n_in => n_in, tanh))

    icnf = ContinuousNormalizingFlows.construct(
        ContinuousNormalizingFlows.ICNF,
        nn,
        nvars,
        naugs;
        compute_mode = ContinuousNormalizingFlows.LuxVecJacMatrixMode(ADTypes.AutoZygote()),
        tspan = (0.0f0, 1.0f0),
        steer_rate = 1.0f-1,
        λ₁ = 1.0f-2,
        λ₂ = 1.0f-2,
        λ₃ = 1.0f-2,
        sol_kwargs = (;
            save_everystep = false,
            alg = OrdinaryDiffEqDefault.DefaultODEAlgorithm(),
            sensealg = SciMLSensitivity.InterpolatingAdjoint(),
        ),
    )
    ps, st = LuxCore.setup(icnf.rng, icnf)
    ps = ComponentArrays.ComponentArray(ps)
    r = rand(icnf.rng, Float32, nvars, n)

    ContinuousNormalizingFlows.loss(icnf, ContinuousNormalizingFlows.TrainMode(), r, ps, st)
    JET.test_call(
        ContinuousNormalizingFlows.loss,
        Base.typesof(icnf, ContinuousNormalizingFlows.TrainMode(), r, ps, st);
        target_modules = [ContinuousNormalizingFlows],
        mode = :typo,
    )
    JET.test_opt(
        ContinuousNormalizingFlows.loss,
        Base.typesof(icnf, ContinuousNormalizingFlows.TrainMode(), r, ps, st);
        target_modules = [ContinuousNormalizingFlows],
    )
end

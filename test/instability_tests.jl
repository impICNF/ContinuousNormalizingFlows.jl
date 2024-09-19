Test.@testset "Instability" begin
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
        ContinuousNormalizingFlows.RNODE,
        nn,
        nvars,
        naugs;
        compute_mode = ContinuousNormalizingFlows.DIJacVecMatrixMode(
            ADTypes.AutoEnzyme(; function_annotation = Enzyme.Const),
        ),
        tspan = (0.0f0, 13.0f0),
        steer_rate = 1.0f-1,
        λ₃ = 1.0f-2,
    )
    ps, st = Lux.setup(icnf.rng, icnf)
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
    )
end

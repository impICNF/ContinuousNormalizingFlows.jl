@testset "Instability" begin
    JET.test_package(
        ContinuousNormalizingFlows;
        target_modules = [ContinuousNormalizingFlows],
        mode = :sound,
    )

    nvars = 2
    r = rand(Float32, nvars)
    nn = Lux.Dense(nvars => nvars, tanh)
    icnf = construct(
        RNODE,
        nn,
        nvars;
        compute_mode = ZygoteVectorMode,
        sol_kwargs = ContinuousNormalizingFlows.sol_kwargs_defaults.medium_noad,
    )
    ps, st = Lux.setup(icnf.rng, icnf)
    ps = ComponentArrays.ComponentArray(ps)
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

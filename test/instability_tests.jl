@testset "Instability" begin
    JET.test_package("ContinuousNormalizingFlows"; target_defined_modules = true)

    nvars = 2
    r = rand(Float32, nvars)
    nn = Lux.Dense(nvars => nvars, tanh)
    icnf = construct(
        RNODE,
        nn,
        nvars;
        sol_kwargs = ContinuousNormalizingFlows.sol_kwargs_defaults.medium_noad,
    )
    ps, st = Lux.setup(icnf.rng, icnf)
    ps = ComponentArrays.ComponentArray(ps)
    loss(icnf, TrainMode(), r, ps, st)
    JET.@test_call target_defined_modules = true loss(icnf, TrainMode(), r, ps, st)
    JET.@test_opt target_defined_modules = true loss(icnf, TrainMode(), r, ps, st)
end

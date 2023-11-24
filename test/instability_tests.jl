@testset "Instability" begin
    JET.test_package("ContinuousNormalizingFlows")

    n = 3
    nvars = 7
    r = rand(Float32, nvars, n)
    nn = Lux.Dense(nvars => nvars, tanh)
    icnf = construct(RNODE, nn, nvars; compute_mode = ZygoteMatrixMode)
    ps, st = Lux.setup(icnf.rng, icnf)
    ps = ComponentArrays.ComponentArray(ps)
    loss(icnf, TrainMode(), r, ps, st)
    JET.@test_call loss(icnf, TrainMode(), r, ps, st)
    JET.@test_opt loss(icnf, TrainMode(), r, ps, st)
end

using ContinuousNormalizingFlows,
    BenchmarkTools,
    ComponentArrays,
    Flux,
    Lux,
    PkgBenchmark,
    Random,
    SciMLSensitivity,
    Zygote

SUITE = BenchmarkGroup()

SUITE["main"] = BenchmarkGroup(["package", "simple"])

SUITE["main"]["Flux"] = BenchmarkGroup(["Flux"])
SUITE["main"]["Lux"] = BenchmarkGroup(["Lux"])

SUITE["main"]["Flux"]["direct"] = BenchmarkGroup(["direct"])
SUITE["main"]["Flux"]["AD-1-order"] = BenchmarkGroup(["gradient"])

SUITE["main"]["Lux"]["direct"] = BenchmarkGroup(["direct"])
SUITE["main"]["Lux"]["AD-1-order"] = BenchmarkGroup(["gradient"])

nvars = 8
n = 128
r = rand(Float32, nvars, n)

nn = FluxCompatLayer(Flux.f32(Flux.Dense(nvars => nvars, tanh)))
icnf = construct(RNODE, nn, nvars; compute_mode = ZygoteMatrixMode)
ps, st = Lux.setup(icnf.rng, icnf)

loss(icnf, TrainMode(), r, ps, st)
loss(icnf, TestMode(), r, ps, st)
Zygote.gradient(loss, icnf, TrainMode(), r, ps, st)
Zygote.gradient(loss, icnf, TestMode(), r, ps, st)
GC.gc()

SUITE["main"]["Flux"]["direct"]["train"] = @benchmarkable loss(icnf, TrainMode(), r, ps, st)
SUITE["main"]["Flux"]["direct"]["test"] = @benchmarkable loss(icnf, TestMode(), r, ps, st)
SUITE["main"]["Flux"]["AD-1-order"]["train"] =
    @benchmarkable Zygote.gradient(loss, icnf, TrainMode(), r, ps, st)
SUITE["main"]["Flux"]["AD-1-order"]["test"] =
    @benchmarkable Zygote.gradient(loss, icnf, TestMode(), r, ps, st)

nn2 = Lux.Dense(nvars => nvars, tanh)
icnf2 = construct(RNODE, nn2, nvars; compute_mode = ZygoteMatrixMode)
icnf2.sol_kwargs[:sensealg] = ForwardDiffSensitivity()
ps2, st2 = Lux.setup(icnf2.rng, icnf2)
ps2 = ComponentArray(ps2)

loss(icnf2, TrainMode(), r, ps2, st2)
loss(icnf2, TestMode(), r, ps2, st2)
Zygote.gradient(loss, icnf2, TrainMode(), r, ps2, st2)
Zygote.gradient(loss, icnf2, TestMode(), r, ps2, st2)
GC.gc()

SUITE["main"]["Lux"]["direct"]["train"] =
    @benchmarkable loss(icnf2, TrainMode(), r, ps2, st2)
SUITE["main"]["Lux"]["direct"]["test"] = @benchmarkable loss(icnf2, TestMode(), r, ps2, st2)
SUITE["main"]["Lux"]["AD-1-order"]["train"] =
    @benchmarkable Zygote.gradient(loss, icnf2, TrainMode(), r, ps2, st2)
SUITE["main"]["Lux"]["AD-1-order"]["test"] =
    @benchmarkable Zygote.gradient(loss, icnf2, TestMode(), r, ps2, st2)

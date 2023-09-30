using ContinuousNormalizingFlows, BenchmarkTools, Flux, Lux, PkgBenchmark, Zygote

SUITE = BenchmarkGroup()

SUITE["main"] = BenchmarkGroup(["package", "simple"])

SUITE["main"]["no_inplace"] = BenchmarkGroup(["no_inplace"])
SUITE["main"]["inplace"] = BenchmarkGroup(["inplace"])

SUITE["main"]["no_inplace"]["direct"] = BenchmarkGroup(["direct"])
SUITE["main"]["no_inplace"]["AD-1-order"] = BenchmarkGroup(["gradient"])

SUITE["main"]["inplace"]["direct"] = BenchmarkGroup(["direct"])
SUITE["main"]["inplace"]["AD-1-order"] = BenchmarkGroup(["gradient"])

nvars = 2^3
n = 2^6
r = rand(Float32, nvars, n)

nn = FluxCompatLayer(Flux.f32(Flux.Dense(nvars => nvars, tanh)))
icnf = construct(RNODE, nn, nvars; compute_mode = ZygoteMatrixMode)
ps, st = Lux.setup(icnf.rng, icnf)

loss(icnf, TrainMode(), r, ps, st)
loss(icnf, TestMode(), r, ps, st)
Zygote.gradient(loss, icnf, TrainMode(), r, ps, st)
Zygote.gradient(loss, icnf, TestMode(), r, ps, st)
GC.gc()

SUITE["main"]["no_inplace"]["direct"]["train"] =
    @benchmarkable loss(icnf, TrainMode(), r, ps, st)
SUITE["main"]["no_inplace"]["direct"]["test"] =
    @benchmarkable loss(icnf, TestMode(), r, ps, st)
SUITE["main"]["no_inplace"]["AD-1-order"]["train"] =
    @benchmarkable Zygote.gradient(loss, icnf, TrainMode(), r, ps, st)
SUITE["main"]["no_inplace"]["AD-1-order"]["test"] =
    @benchmarkable Zygote.gradient(loss, icnf, TestMode(), r, ps, st)

nn2 = FluxCompatLayer(Flux.f32(Flux.Dense(nvars => nvars, tanh)))
icnf2 = construct(RNODE, nn2, nvars; compute_mode = ZygoteMatrixMode, inplace = true)
ps2, st2 = Lux.setup(icnf2.rng, icnf2)

loss(icnf2, TrainMode(), r, ps2, st2)
loss(icnf2, TestMode(), r, ps2, st2)
Zygote.gradient(loss, icnf2, TrainMode(), r, ps2, st2)
Zygote.gradient(loss, icnf2, TestMode(), r, ps2, st2)
GC.gc()

SUITE["main"]["inplace"]["direct"]["train"] =
    @benchmarkable loss(icnf2, TrainMode(), r, ps2, st2)
SUITE["main"]["inplace"]["direct"]["test"] =
    @benchmarkable loss(icnf2, TestMode(), r, ps2, st2)
SUITE["main"]["inplace"]["AD-1-order"]["train"] =
    @benchmarkable Zygote.gradient(loss, icnf2, TrainMode(), r, ps2, st2)
SUITE["main"]["inplace"]["AD-1-order"]["test"] =
    @benchmarkable Zygote.gradient(loss, icnf2, TestMode(), r, ps2, st2)

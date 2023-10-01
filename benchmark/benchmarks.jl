using ContinuousNormalizingFlows,
    BenchmarkTools, ComponentArrays, Flux, Lux, PkgBenchmark, SciMLSensitivity, Zygote

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
nn = Lux.Dense(nvars => nvars; use_bias = false)

icnf = construct(RNODE, nn, nvars; compute_mode = ZygoteMatrixMode)
icnf.sol_kwargs[:sensealg] = ForwardDiffSensitivity()
icnf.sol_kwargs[:verbose] = true
ps, st = Lux.setup(icnf.rng, icnf)
ps = ComponentArray(ps)

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

icnf2 = construct(RNODE, nn, nvars; compute_mode = ZygoteMatrixMode, inplace = true)
icnf2.sol_kwargs[:sensealg] = ForwardDiffSensitivity()
icnf2.sol_kwargs[:verbose] = true

loss(icnf2, TrainMode(), r, ps, st)
loss(icnf2, TestMode(), r, ps, st)
Zygote.gradient(loss, icnf2, TrainMode(), r, ps, st)
Zygote.gradient(loss, icnf2, TestMode(), r, ps, st)
GC.gc()

SUITE["main"]["inplace"]["direct"]["train"] =
    @benchmarkable loss(icnf2, TrainMode(), r, ps, st)
SUITE["main"]["inplace"]["direct"]["test"] =
    @benchmarkable loss(icnf2, TestMode(), r, ps, st)
SUITE["main"]["inplace"]["AD-1-order"]["train"] =
    @benchmarkable Zygote.gradient(loss, icnf2, TrainMode(), r, ps, st)
SUITE["main"]["inplace"]["AD-1-order"]["test"] =
    @benchmarkable Zygote.gradient(loss, icnf2, TestMode(), r, ps, st)

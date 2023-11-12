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
nn = Lux.Dense(nvars => nvars, tanh)

icnf = construct(
    RNODE,
    nn,
    nvars;
    compute_mode = ZygoteMatrixMode,
    sol_kwargs = merge(
        ContinuousNormalizingFlows.sol_kwargs_defaults.medium,
        (sensealg = ForwardDiffSensitivity(),),
    ),
)
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

icnf2 = construct(
    RNODE,
    nn,
    nvars;
    compute_mode = ZygoteMatrixMode,
    inplace = true,
    sol_kwargs = merge(
        ContinuousNormalizingFlows.sol_kwargs_defaults.medium,
        (sensealg = ForwardDiffSensitivity(),),
    ),
)

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

using ContinuousNormalizingFlows,
    BenchmarkTools, ComponentArrays, Lux, PkgBenchmark, StableRNGs, Zygote

SUITE = BenchmarkGroup()

SUITE["main"] = BenchmarkGroup(["package", "simple"])

SUITE["main"]["no_inplace"] = BenchmarkGroup(["no_inplace"])
SUITE["main"]["inplace"] = BenchmarkGroup(["inplace"])

SUITE["main"]["no_inplace"]["direct"] = BenchmarkGroup(["direct"])
SUITE["main"]["no_inplace"]["AD-1-order"] = BenchmarkGroup(["gradient"])

SUITE["main"]["inplace"]["direct"] = BenchmarkGroup(["direct"])
SUITE["main"]["inplace"]["AD-1-order"] = BenchmarkGroup(["gradient"])

rng = StableRNG(12345)
nvars = 2^3
r = rand(Float32, nvars)
nn = Dense(nvars => nvars, tanh)

icnf = construct(
    RNODE,
    nn,
    nvars;
    sol_kwargs = ContinuousNormalizingFlows.sol_kwargs_defaults.medium_noad,
    rng,
)
ps, st = Lux.setup(icnf.rng, icnf)
ps = ComponentArray(ps)

diff_loss_tn(x) = loss(icnf, TrainMode(), r, x, st)
diff_loss_tt(x) = loss(icnf, TestMode(), r, x, st)

diff_loss_tn(ps)
diff_loss_tt(ps)
Zygote.gradient(diff_loss_tn, ps)
Zygote.gradient(diff_loss_tt, ps)
GC.gc()

SUITE["main"]["no_inplace"]["direct"]["train"] = @benchmarkable diff_loss_tn(ps)
SUITE["main"]["no_inplace"]["direct"]["test"] = @benchmarkable diff_loss_tt(ps)
SUITE["main"]["no_inplace"]["AD-1-order"]["train"] =
    @benchmarkable Zygote.gradient(diff_loss_tn, ps)
SUITE["main"]["no_inplace"]["AD-1-order"]["test"] =
    @benchmarkable Zygote.gradient(diff_loss_tt, ps)

icnf2 = construct(
    RNODE,
    nn,
    nvars;
    inplace = true,
    sol_kwargs = ContinuousNormalizingFlows.sol_kwargs_defaults.medium_noad,
    rng,
)

diff_loss_tn2(x) = loss(icnf2, TrainMode(), r, x, st)
diff_loss_tt2(x) = loss(icnf2, TestMode(), r, x, st)

diff_loss_tn2(ps)
diff_loss_tt2(ps)
Zygote.gradient(diff_loss_tn2, ps)
Zygote.gradient(diff_loss_tt2, ps)
GC.gc()

SUITE["main"]["inplace"]["direct"]["train"] = @benchmarkable diff_loss_tn2(ps)
SUITE["main"]["inplace"]["direct"]["test"] = @benchmarkable diff_loss_tt2(ps)
SUITE["main"]["inplace"]["AD-1-order"]["train"] =
    @benchmarkable Zygote.gradient(diff_loss_tn2, ps)
SUITE["main"]["inplace"]["AD-1-order"]["test"] =
    @benchmarkable Zygote.gradient(diff_loss_tt2, ps)

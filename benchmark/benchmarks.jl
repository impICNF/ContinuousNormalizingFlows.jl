using ContinuousNormalizingFlows,
    ADTypes,
    BenchmarkTools,
    ComponentArrays,
    DifferentiationInterface,
    Lux,
    PkgBenchmark,
    StableRNGs,
    Zygote

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
naugs = nvars
n_in = nvars + naugs
n = 2^6
nn = Dense(n_in => n_in, tanh)

icnf = construct(
    RNODE,
    nn,
    nvars,
    naugs;
    compute_mode = ZygoteMatrixMode,
    tspan = (0.0f0, 13.0f0),
    steer_rate = 1.0f-1,
    λ₃ = 1.0f-2,
    rng,
)
ps, st = Lux.setup(icnf.rng, icnf)
ps = ComponentArray(ps)
r = rand(icnf.rng, Float32, nvars, n)

diff_loss_tn(x) = loss(icnf, TrainMode(), r, x, st)
diff_loss_tt(x) = loss(icnf, TestMode(), r, x, st)

diff_loss_tn(ps)
diff_loss_tt(ps)
DifferentiationInterface.gradient(diff_loss_tn, AutoZygote(), ps)
DifferentiationInterface.gradient(diff_loss_tt, AutoZygote(), ps)
GC.gc()

SUITE["main"]["no_inplace"]["direct"]["train"] = @benchmarkable diff_loss_tn(ps)
SUITE["main"]["no_inplace"]["direct"]["test"] = @benchmarkable diff_loss_tt(ps)
SUITE["main"]["no_inplace"]["AD-1-order"]["train"] =
    @benchmarkable DifferentiationInterface.gradient(diff_loss_tn, AutoZygote(), ps)
SUITE["main"]["no_inplace"]["AD-1-order"]["test"] =
    @benchmarkable DifferentiationInterface.gradient(diff_loss_tt, AutoZygote(), ps)

icnf2 = construct(
    RNODE,
    nn,
    nvars,
    naugs;
    inplace = true,
    compute_mode = ZygoteMatrixMode,
    tspan = (0.0f0, 13.0f0),
    steer_rate = 1.0f-1,
    λ₃ = 1.0f-2,
    rng,
)

diff_loss_tn2(x) = loss(icnf2, TrainMode(), r, x, st)
diff_loss_tt2(x) = loss(icnf2, TestMode(), r, x, st)

diff_loss_tn2(ps)
diff_loss_tt2(ps)
DifferentiationInterface.gradient(diff_loss_tn2, AutoZygote(), ps)
DifferentiationInterface.gradient(diff_loss_tt2, AutoZygote(), ps)
GC.gc()

SUITE["main"]["inplace"]["direct"]["train"] = @benchmarkable diff_loss_tn2(ps)
SUITE["main"]["inplace"]["direct"]["test"] = @benchmarkable diff_loss_tt2(ps)
SUITE["main"]["inplace"]["AD-1-order"]["train"] =
    @benchmarkable DifferentiationInterface.gradient(diff_loss_tn2, AutoZygote(), ps)
SUITE["main"]["inplace"]["AD-1-order"]["test"] =
    @benchmarkable DifferentiationInterface.gradient(diff_loss_tt2, AutoZygote(), ps)

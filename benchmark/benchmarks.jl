import ADTypes,
    BenchmarkTools,
    ComponentArrays,
    DifferentiationInterface,
    Distributions,
    ForwardDiff,
    Lux,
    LuxCore,
    PkgBenchmark,
    StableRNGs,
    Zygote,
    ContinuousNormalizingFlows

rng = StableRNGs.StableRNG(1)
ndata = 2^10
ndimension = 1
data_dist = Distributions.Beta{Float32}(2.0f0, 4.0f0)
r = rand(rng, data_dist, ndimension, ndata)
r = convert.(Float32, r)

nvars = size(r, 1)
naugs = nvars + 1
n_in = nvars + naugs

nn = Lux.Chain(
    Lux.Dense(n_in => (2 * n_in + 1), tanh),
    Lux.Dense((2 * n_in + 1) => n_in, tanh),
)

icnf = ContinuousNormalizingFlows.ICNF(; nvars, naugmented = naugs, nn, rng)

icnf2 =
    ContinuousNormalizingFlows.ICNF(; nvars, naugmented = naugs, nn, inplace = true, rng)

ps, st = LuxCore.setup(icnf.rng, icnf)
ps = ComponentArrays.ComponentArray(ps)

function diff_loss_tn(x::Any)
    return ContinuousNormalizingFlows.loss(
        icnf,
        ContinuousNormalizingFlows.TrainMode{true}(),
        r,
        x,
        st,
    )
end
function diff_loss_tt(x::Any)
    return ContinuousNormalizingFlows.loss(
        icnf,
        ContinuousNormalizingFlows.TestMode{true}(),
        r,
        x,
        st,
    )
end

function diff_loss_tn2(x::Any)
    return ContinuousNormalizingFlows.loss(
        icnf2,
        ContinuousNormalizingFlows.TrainMode{true}(),
        r,
        x,
        st,
    )
end
function diff_loss_tt2(x::Any)
    return ContinuousNormalizingFlows.loss(
        icnf2,
        ContinuousNormalizingFlows.TestMode{true}(),
        r,
        x,
        st,
    )
end

diff_loss_tn(ps)
diff_loss_tt(ps)
DifferentiationInterface.gradient(diff_loss_tn, ADTypes.AutoZygote(), ps)
DifferentiationInterface.gradient(diff_loss_tt, ADTypes.AutoZygote(), ps)

diff_loss_tn2(ps)
diff_loss_tt2(ps)
DifferentiationInterface.gradient(diff_loss_tn2, ADTypes.AutoZygote(), ps)
DifferentiationInterface.gradient(diff_loss_tt2, ADTypes.AutoZygote(), ps)

GC.gc()

SUITE = BenchmarkTools.BenchmarkGroup()

SUITE["main"] = BenchmarkTools.BenchmarkGroup(["package", "simple"])

SUITE["main"]["no_inplace"] = BenchmarkTools.BenchmarkGroup(["no_inplace"])
SUITE["main"]["inplace"] = BenchmarkTools.BenchmarkGroup(["inplace"])

SUITE["main"]["no_inplace"]["direct"] = BenchmarkTools.BenchmarkGroup(["direct"])
SUITE["main"]["no_inplace"]["AD-1-order"] = BenchmarkTools.BenchmarkGroup(["gradient"])

SUITE["main"]["inplace"]["direct"] = BenchmarkTools.BenchmarkGroup(["direct"])
SUITE["main"]["inplace"]["AD-1-order"] = BenchmarkTools.BenchmarkGroup(["gradient"])

SUITE["main"]["no_inplace"]["direct"]["train"] =
    BenchmarkTools.@benchmarkable diff_loss_tn(ps)
SUITE["main"]["no_inplace"]["direct"]["test"] =
    BenchmarkTools.@benchmarkable diff_loss_tt(ps)
SUITE["main"]["no_inplace"]["AD-1-order"]["train"] =
    BenchmarkTools.@benchmarkable DifferentiationInterface.gradient(
        diff_loss_tn,
        ADTypes.AutoZygote(),
        ps,
    )
SUITE["main"]["no_inplace"]["AD-1-order"]["test"] =
    BenchmarkTools.@benchmarkable DifferentiationInterface.gradient(
        diff_loss_tt,
        ADTypes.AutoZygote(),
        ps,
    )
SUITE["main"]["inplace"]["direct"]["train"] =
    BenchmarkTools.@benchmarkable diff_loss_tn2(ps)
SUITE["main"]["inplace"]["direct"]["test"] = BenchmarkTools.@benchmarkable diff_loss_tt2(ps)
SUITE["main"]["inplace"]["AD-1-order"]["train"] =
    BenchmarkTools.@benchmarkable DifferentiationInterface.gradient(
        diff_loss_tn2,
        ADTypes.AutoZygote(),
        ps,
    )
SUITE["main"]["inplace"]["AD-1-order"]["test"] =
    BenchmarkTools.@benchmarkable DifferentiationInterface.gradient(
        diff_loss_tt2,
        ADTypes.AutoZygote(),
        ps,
    )

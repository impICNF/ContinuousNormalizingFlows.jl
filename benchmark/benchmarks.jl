import ADTypes,
    BenchmarkTools,
    ComponentArrays,
    DifferentiationInterface,
    Lux,
    OrdinaryDiffEqAdamsBashforthMoulton,
    PkgBenchmark,
    SciMLSensitivity,
    StableRNGs,
    Static,
    Zygote,
    ContinuousNormalizingFlows

SUITE = BenchmarkTools.BenchmarkGroup()

SUITE["main"] = BenchmarkTools.BenchmarkGroup(["package", "simple"])

SUITE["main"]["no_inplace"] = BenchmarkTools.BenchmarkGroup(["no_inplace"])
SUITE["main"]["inplace"] = BenchmarkTools.BenchmarkGroup(["inplace"])

SUITE["main"]["no_inplace"]["direct"] = BenchmarkTools.BenchmarkGroup(["direct"])
SUITE["main"]["no_inplace"]["AD-1-order"] = BenchmarkTools.BenchmarkGroup(["gradient"])

SUITE["main"]["inplace"]["direct"] = BenchmarkTools.BenchmarkGroup(["direct"])
SUITE["main"]["inplace"]["AD-1-order"] = BenchmarkTools.BenchmarkGroup(["gradient"])

rng = StableRNGs.StableRNG(1)
nvars = 2^3
naugs = nvars
n_in = nvars + naugs
n = 2^6
nn = Lux.Chain(Lux.Dense(n_in => n_in, tanh))

icnf = ContinuousNormalizingFlows.construct(
    ContinuousNormalizingFlows.ICNF,
    nn,
    nvars,
    naugs;
    compute_mode = ContinuousNormalizingFlows.LuxVecJacMatrixMode(ADTypes.AutoZygote()),
    tspan = (0.0f0, 1.0f0),
    steer_rate = 1.0f-1,
    λ₁ = 1.0f-2,
    λ₂ = 1.0f-2,
    λ₃ = 1.0f-2,
    rng,
    sol_kwargs = (;
        progress = true,
        save_everystep = false,
        reltol = sqrt(eps(one(Float32))),
        abstol = eps(one(Float32)),
        maxiters = typemax(Int),
        alg = OrdinaryDiffEqAdamsBashforthMoulton.VCABM(; thread = Static.True()),
        sensealg = SciMLSensitivity.InterpolatingAdjoint(;
            autodiff = true,
            autojacvec = ZygoteVJP(),
            checkpointing = true,
        ),
    ),
)
ps, st = Lux.setup(icnf.rng, icnf)
ps = ComponentArrays.ComponentArray(ps)
r = rand(icnf.rng, Float32, nvars, n)

function diff_loss_tn(x::Any)
    return ContinuousNormalizingFlows.loss(
        icnf,
        ContinuousNormalizingFlows.TrainMode(),
        r,
        x,
        st,
    )
end
function diff_loss_tt(x::Any)
    return ContinuousNormalizingFlows.loss(
        icnf,
        ContinuousNormalizingFlows.TestMode(),
        r,
        x,
        st,
    )
end

diff_loss_tn(ps)
diff_loss_tt(ps)
DifferentiationInterface.gradient(diff_loss_tn, ADTypes.AutoZygote(), ps)
DifferentiationInterface.gradient(diff_loss_tt, ADTypes.AutoZygote(), ps)
GC.gc()

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

icnf2 = ContinuousNormalizingFlows.construct(
    ContinuousNormalizingFlows.ICNF,
    nn,
    nvars,
    naugs;
    inplace = true,
    compute_mode = ContinuousNormalizingFlows.LuxVecJacMatrixMode(ADTypes.AutoZygote()),
    tspan = (0.0f0, 1.0f0),
    steer_rate = 1.0f-1,
    λ₁ = 1.0f-2,
    λ₂ = 1.0f-2,
    λ₃ = 1.0f-2,
    rng,
    sol_kwargs = (;
        progress = true,
        save_everystep = false,
        reltol = sqrt(eps(one(Float32))),
        abstol = eps(one(Float32)),
        maxiters = typemax(Int),
        alg = OrdinaryDiffEqAdamsBashforthMoulton.VCABM(; thread = Static.True()),
        sensealg = SciMLSensitivity.InterpolatingAdjoint(;
            autodiff = true,
            autojacvec = ZygoteVJP(),
            checkpointing = true,
        ),
    ),
)

function diff_loss_tn2(x::Any)
    return ContinuousNormalizingFlows.loss(
        icnf2,
        ContinuousNormalizingFlows.TrainMode(),
        r,
        x,
        st,
    )
end
function diff_loss_tt2(x::Any)
    return ContinuousNormalizingFlows.loss(
        icnf2,
        ContinuousNormalizingFlows.TestMode(),
        r,
        x,
        st,
    )
end

diff_loss_tn2(ps)
diff_loss_tt2(ps)
DifferentiationInterface.gradient(diff_loss_tn2, ADTypes.AutoZygote(), ps)
DifferentiationInterface.gradient(diff_loss_tt2, ADTypes.AutoZygote(), ps)
GC.gc()

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

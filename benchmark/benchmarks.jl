using ContinuousNormalizingFlows, BenchmarkTools, PkgBenchmark

SUITE = BenchmarkGroup()

SUITE["main"] = BenchmarkGroup(["package", "simple"])
SUITE["main"]["0-order"] = BenchmarkGroup(["direct"])
SUITE["main"]["1-order"] = BenchmarkGroup(["gradient"])

BenchmarkTools.DEFAULT_PARAMETERS.samples = 2^13
BenchmarkTools.DEFAULT_PARAMETERS.seconds = convert(Float64, 2 * 60)
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1
BenchmarkTools.DEFAULT_PARAMETERS.overhead = 0.0
BenchmarkTools.DEFAULT_PARAMETERS.gctrial = true
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = false
BenchmarkTools.DEFAULT_PARAMETERS.time_tolerance = convert(Float64, eps(Float16))
BenchmarkTools.DEFAULT_PARAMETERS.memory_tolerance = convert(Float64, eps(Float16))

nvars = 8
n = 128
rng = Random.default_rng()
r = rand(Float32, nvars, n)

nn = FluxCompatLayer(Flux.f32(Flux.Dense(nvars => nvars, tanh)))
icnf = construct(RNODE, nn, nvars; compute_mode = ZygoteMatrixMode)

ps, st = Lux.setup(rng, icnf)
diff_loss_train(x) = loss(icnf, TrainMode(), r, x, st)
diff_loss_test(x) = loss(icnf, TestMode(), r, x, st)
grad_diff_loss_train() = Zygote.gradient(diff_loss_train, ps)
grad_diff_loss_test() = Zygote.gradient(diff_loss_test, ps)
t_loss_train() = loss(icnf, TrainMode(), r, ps, st)
t_loss_test() = loss(icnf, TestMode(), r, ps, st)

SUITE["main"]["0-order"]["train"] = @benchmarkable $t_loss_train()
SUITE["main"]["0-order"]["test"] = @benchmarkable $t_loss_test()
SUITE["main"]["1-order"]["train"] = @benchmarkable $grad_diff_loss_train()
SUITE["main"]["1-order"]["test"] = @benchmarkable $grad_diff_loss_test()

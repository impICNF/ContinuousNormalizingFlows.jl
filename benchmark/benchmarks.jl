using ContinuousNormalizingFlows, BenchmarkTools, Flux, Lux, PkgBenchmark, Random, Zygote

SUITE = BenchmarkGroup()

SUITE["main"] = BenchmarkGroup(["package", "simple"])
SUITE["main"]["0-order"] = BenchmarkGroup(["direct"])
SUITE["main"]["1-order"] = BenchmarkGroup(["gradient"])

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

t_loss_train()
t_loss_test()
grad_diff_loss_train()
grad_diff_loss_test()
GC.gc()

SUITE["main"]["0-order"]["train"] = @benchmarkable $t_loss_train()
SUITE["main"]["0-order"]["test"] = @benchmarkable $t_loss_test()
SUITE["main"]["1-order"]["train"] = @benchmarkable $grad_diff_loss_train()
SUITE["main"]["1-order"]["test"] = @benchmarkable $grad_diff_loss_test()

@testset "Benchmark" begin
    nvars = 8
    n = 128
    rng = Random.default_rng()
    r = rand(Float32, nvars, n)

    nn = FluxCompatLayer(Flux.f32(Flux.Dense(nvars => nvars, tanh)))
    icnf = construct(RNODE, nn, nvars; compute_mode = ZygoteMatrixMode)

    ps, st = Lux.setup(rng, icnf)
    diff_loss_train(x) = loss(icnf, r, x, st; mode = TrainMode())
    diff_loss_test(x) = loss(icnf, r, x, st; mode = TestMode())
    grad_diff_loss_train() = Zygote.gradient(diff_loss_train, ps)
    grad_diff_loss_test() = Zygote.gradient(diff_loss_test, ps)
    t_loss_train() = loss(icnf, r, ps, st; mode = TrainMode())
    t_loss_test() = loss(icnf, r, ps, st; mode = TestMode())

    ben_loss_train = BenchmarkTools.@benchmark $t_loss_train()
    ben_loss_test = BenchmarkTools.@benchmark $t_loss_test()
    ben_grad_train = BenchmarkTools.@benchmark $grad_diff_loss_train()
    ben_grad_test = BenchmarkTools.@benchmark $grad_diff_loss_test()

    @info "t_loss_train"
    display(ben_loss_train)

    @info "t_loss_test"
    display(ben_loss_test)

    @info "grad_diff_loss_train"
    display(ben_grad_train)

    @info "grad_diff_loss_test"
    display(ben_grad_test)

    @test true
end

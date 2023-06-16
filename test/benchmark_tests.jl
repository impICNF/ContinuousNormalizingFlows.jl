@testset "Benchmark" begin
    nvars = 8
    n = 128
    rng = Random.default_rng()
    r = rand(Float32, nvars, n)

    nn = FluxCompatLayer(Flux.f32(Flux.Dense(nvars => nvars, tanh)))
    icnf = construct(RNODE, nn, nvars; compute_mode = ZygoteMatrixMode)

    ps, st = Lux.setup(rng, icnf)
    diff_loss(x) = loss(icnf, r, x, st)
    grad_diff_loss() = Zygote.gradient(diff_loss, ps)
    t_loss() = loss(icnf, r, ps, st)

    ben_1 = BenchmarkTools.@benchmark $t_loss()
    ben_2 = BenchmarkTools.@benchmark $grad_diff_loss()

    @info "t_loss"
    display(ben_1)

    @info "grad_diff_loss"
    display(ben_2)

    @test true
end

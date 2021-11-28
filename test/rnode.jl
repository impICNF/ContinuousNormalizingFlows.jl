@testset "RNODE" begin
    @testset "$cr-$tp-$nvars-Var-Smoke-Test" for
            cr in [CPU1(), CUDALibs()],
            tp in [Float64, Float32, Float16],
            nvars in 1:3
        icnf = RNODE{tp}(Dense(nvars, nvars), nvars; acceleration=cr)
        ufd = copy(icnf.p)
        n = 8
        r = rand(tp, nvars, n)

        @test !isnothing(inference(icnf, TestMode(), r))
        @test !isnothing(inference(icnf, TrainMode(), r))

        @test !isnothing(generate(icnf, TestMode(), n))
        @test !isnothing(generate(icnf, TrainMode(), n))

        @test !isnothing(icnf(r))
        @test !isnothing(loss_f(icnf)(r))

        d = ICNFDistribution(; m=icnf)

        @test !isnothing(logpdf(d, r))
        @test !isnothing(pdf(d, r))
        @test !isnothing(rand(d, n))

        df = DataFrame(r', :auto)
        model = ICNFModel(; m=icnf, n_epochs=8)
        mach = machine(model, df)
        fit!(mach)
        fd = MLJBase.fitted_params(mach).learned_parameters

        @test !isnothing(MLJBase.transform(mach, df))
        @test fd != ufd
    end
end

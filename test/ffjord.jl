@testset "FFJORD" begin
    @testset "$cr-$tp-$nvars-Var-Smoke-Test" for
            cr in [CPU1(), CUDALibs()],
            tp in [Float64, Float32, Float16],
            nvars in 1:3
        ffjord = FFJORD{tp}(Dense(nvars, nvars), nvars; acceleration=cr)
        ufd = copy(ffjord.p)
        n = 8
        r = rand(tp, nvars, n)

        @test !isnothing(inference(ffjord, TestMode(), r))
        @test !isnothing(inference(ffjord, TrainMode(), r))

        @test !isnothing(generate(ffjord, TestMode(), n))
        @test !isnothing(generate(ffjord, TrainMode(), n))

        @test !isnothing(ffjord(r))
        @test !isnothing(loss_f(ffjord)(r))

        d = ICNFDistribution(; m=ffjord)

        @test !isnothing(logpdf(d, r))
        @test !isnothing(pdf(d, r))
        @test !isnothing(rand(d, n))

        df = DataFrame(r', :auto)
        model = ICNFModel(; m=ffjord)
        mach = machine(model, df)
        fit!(mach)
        fd = MLJBase.fitted_params(mach).learned_parameters

        @test !isnothing(MLJBase.transform(mach, df))
        @test fd != ufd

        @test !isnothing(ICNFModel(nvars))
    end
end

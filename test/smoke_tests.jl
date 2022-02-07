@testset "Smoke Tests" begin
    mts = UnionAll[FFJORD, RNODE, Planar]
    cmts = UnionAll[CondFFJORD, CondRNODE, CondPlanar]
    crs = AbstractResource[CPU1()]
    if has_cuda_gpu()
        push!(crs, CUDALibs())
    end
    tps = DataType[Float64, Float32, Float16]
    nvars_ = 1:3
    n_epochs = 8

    @testset "$mt | $cr | $tp | $nvars Vars" for
            mt in mts,
            cr in crs,
            tp in tps,
            nvars in nvars_
        if mt <: Planar
            nn = PlanarNN{tp}(nvars, tanh)
        else
            nn = Chain(
                Dense(nvars, nvars, tanh),
            )
        end
        icnf = mt{tp}(nn, nvars; acceleration=cr)
        ufd = copy(icnf.p)
        n = 8
        r = rand(tp, nvars, n)

        @test !isnothing(inference(icnf, TestMode(), r))
        @test !isnothing(inference(icnf, TrainMode(), r))

        @test !isnothing(generate(icnf, TestMode(), n))
        @test !isnothing(generate(icnf, TrainMode(), n))

        @test !isnothing(icnf(r))
        @test !isnothing(loss_f(icnf)(r))

        @test !isnothing(agg_loglikelihood(icnf, r))

        d = ICNFDistribution(icnf)

        @test !isnothing(logpdf(d, r))
        @test !isnothing(pdf(d, r))
        @test !isnothing(rand(d))
        @test !isnothing(rand(d, n))

        df = DataFrame(r', :auto)
        model = ICNFModel(icnf; n_epochs)
        mach = machine(model, df)
        @test !isnothing(fit!(mach))
        fd = MLJBase.fitted_params(mach).learned_parameters
        @test !isnothing(MLJBase.transform(mach, df))

        if tp <: Float16
            @test_broken fd != ufd
        else
            @test fd != ufd
        end
    end
    @testset "$mt | $cr | $tp | $nvars Vars" for
            mt in cmts,
            cr in crs,
            tp in tps,
            nvars in nvars_
        if mt <: CondPlanar
            nn = PlanarNN{tp}(nvars, tanh; cond=true)
        else
            nn = Chain(
                Dense(nvars*2, nvars, tanh),
            )
        end
        icnf = mt{tp}(nn, nvars; acceleration=cr)
        ufd = copy(icnf.p)
        n = 8
        r = rand(tp, nvars, n)
        r2 = rand(tp, nvars, n)

        @test !isnothing(inference(icnf, TestMode(), r, r2))
        @test !isnothing(inference(icnf, TrainMode(), r, r2))

        @test !isnothing(generate(icnf, TestMode(), r2, n))
        @test !isnothing(generate(icnf, TrainMode(), r2, n))

        @test !isnothing(icnf(r, r2))
        @test !isnothing(loss_f(icnf)(r, r2))

        @test !isnothing(agg_loglikelihood(icnf, r, r2))

        df = DataFrame(r', :auto)
        df2 = DataFrame(r2', :auto)
        model = CondICNFModel(icnf; n_epochs)
        mach = machine(model, (df, df2))
        @test !isnothing(fit!(mach))
        fd = MLJBase.fitted_params(mach).learned_parameters
        @test !isnothing(MLJBase.transform(mach, (df, df2)))

        if tp <: Float16
            @test_broken fd != ufd
        else
            @test fd != ufd
        end
    end
end

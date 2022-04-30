@testset "Smoke Tests" begin
    mts = UnionAll[FFJORD, RNODE, Planar]
    mts_fit = UnionAll[RNODE]
    cmts = UnionAll[CondFFJORD, CondRNODE, CondPlanar]
    cmts_fit = UnionAll[CondRNODE]
    crs = AbstractResource[CPU1()]
    if has_cuda_gpu()
        push!(crs, CUDALibs())
    end
    tps = DataType[Float64, Float32, Float16]
    tps_fit = DataType[Float32]
    nvars_ = (1:2)
    n_epochs = 8
    batch_size = 8
    n = 8*4

    @testset "$mt | $cr | $tp | $nvars Vars" for
            mt in mts,
            cr in crs,
            tp in tps,
            nvars in nvars_
        data_dist = Beta{tp}(convert.(tp, (2, 4))...)
        r = convert.(tp, rand(data_dist, nvars, n))

        if mt <: Planar
            nn = PlanarNN(nvars, tanh)
        else
            nn = Chain(
                Dense(nvars, nvars, tanh),
            )
        end
        icnf = mt{tp}(nn, nvars; acceleration=cr)
        ufd = copy(icnf.p)

        @test !isnothing(inference(icnf, TestMode(), r))
        @test !isnothing(inference(icnf, TrainMode(), r))
        @test !isnothing(generate(icnf, TestMode(), n))
        @test !isnothing(generate(icnf, TrainMode(), n))

        @test !isnothing(icnf(r))
        @test !isnothing(loss(icnf, r))
        @test !isnothing(loss_f(icnf, FluxOptApp())(r))
        @test !isnothing(loss_f(icnf, OptimOptApp(), [(r,), nothing])(icnf.p))
        @test !isnothing(loss_f(icnf, SciMLOptApp())(icnf.p, SciMLBase.NullParameters(), r))

        @test !isnothing(agg_loglikelihood(icnf, r))

        d = ICNFDistribution(icnf)

        @test !isnothing(logpdf(d, r))
        @test !isnothing(pdf(d, r))
        @test !isnothing(rand(d))
        @test !isnothing(rand(d, n))
    end
    @testset "Fitting | $mt | $cr | $tp | $nvars Vars" for
            mt in mts_fit,
            cr in crs,
            tp in tps_fit,
            nvars in nvars_
        data_dist = Beta{tp}(convert.(tp, (2, 4))...)
        r = convert.(tp, rand(data_dist, nvars, n))
        df = DataFrame(r', :auto)

        # Flux Opt
        if mt <: Planar
            nn = PlanarNN(nvars, tanh)
        else
            nn = Chain(
                Dense(nvars, nvars, tanh),
            )
        end
        icnf = mt{tp}(nn, nvars; acceleration=cr)
        ufd = copy(icnf.p)
        model = ICNFModel(icnf; n_epochs, batch_size, opt_app=FluxOptApp())
        mach = machine(model, df)
        @test !isnothing(fit!(mach))
        fd = MLJBase.fitted_params(mach).learned_parameters
        @test !isnothing(MLJBase.transform(mach, df))
        @test fd != ufd

        # Optim Opt
        if mt <: Planar
            nn = PlanarNN(nvars, tanh)
        else
            nn = Chain(
                Dense(nvars, nvars, tanh),
            )
        end
        icnf = mt{tp}(nn, nvars; acceleration=cr)
        ufd = copy(icnf.p)
        model = ICNFModel(icnf; n_epochs, batch_size, opt_app=OptimOptApp())
        mach = machine(model, df)
        @test !isnothing(fit!(mach))
        fd = MLJBase.fitted_params(mach).learned_parameters
        @test !isnothing(MLJBase.transform(mach, df))
        @test fd != ufd

        # SciML Opt with Zygote
        if mt <: Planar
            nn = PlanarNN(nvars, tanh)
        else
            nn = Chain(
                Dense(nvars, nvars, tanh),
            )
        end
        icnf = mt{tp}(nn, nvars; acceleration=cr)
        ufd = copy(icnf.p)
        model = ICNFModel(icnf; n_epochs, batch_size, opt_app=SciMLOptApp(), adtype=GalacticOptim.AutoZygote())
        mach = machine(model, df)
        @test !isnothing(fit!(mach))
        fd = MLJBase.fitted_params(mach).learned_parameters
        @test !isnothing(MLJBase.transform(mach, df))
        @test fd != ufd

        # SciML Opt with ForwardDiff
        if mt <: Planar
            nn = PlanarNN(nvars, tanh)
        else
            nn = Chain(
                Dense(nvars, nvars, tanh),
            )
        end
        icnf = mt{tp}(nn, nvars; acceleration=cr)
        ufd = copy(icnf.p)
        model = ICNFModel(icnf; n_epochs, batch_size, opt_app=SciMLOptApp(), adtype=GalacticOptim.AutoForwardDiff())
        mach = machine(model, df)
        @test !isnothing(fit!(mach))
        fd = MLJBase.fitted_params(mach).learned_parameters
        @test !isnothing(MLJBase.transform(mach, df))
        @test fd != ufd
    end
    @testset "$mt | $cr | $tp | $nvars Vars" for
            mt in cmts,
            cr in crs,
            tp in tps,
            nvars in nvars_
        data_dist = Beta{tp}(convert.(tp, (2, 4))...)
        data_dist2 = Beta{tp}(convert.(tp, (4, 2))...)
        r = convert.(tp, rand(data_dist, nvars, n))
        r2 = convert.(tp, rand(data_dist, nvars, n))

        if mt <: CondPlanar
            nn = PlanarNN(nvars, tanh; cond=true)
        else
            nn = Chain(
                Dense(nvars*2, nvars, tanh),
            )
        end
        icnf = mt{tp}(nn, nvars; acceleration=cr)
        ufd = copy(icnf.p)

        @test !isnothing(inference(icnf, TestMode(), r, r2))
        @test !isnothing(inference(icnf, TrainMode(), r, r2))
        @test !isnothing(generate(icnf, TestMode(), r2, n))
        @test !isnothing(generate(icnf, TrainMode(), r2, n))

        @test !isnothing(icnf(r, r2))
        @test !isnothing(loss(icnf, r, r2))
        @test !isnothing(loss_f(icnf, FluxOptApp())(r, r2))
        @test !isnothing(loss_f(icnf, OptimOptApp(), [(r, r2), nothing])(icnf.p))
        @test !isnothing(loss_f(icnf, SciMLOptApp())(icnf.p, SciMLBase.NullParameters(), r, r2))

        @test !isnothing(agg_loglikelihood(icnf, r, r2))
    end
    @testset "Fitting | $mt | $cr | $tp | $nvars Vars" for
            mt in cmts_fit,
            cr in crs,
            tp in tps_fit,
            nvars in nvars_
        data_dist = Beta{tp}(convert.(tp, (2, 4))...)
        data_dist2 = Beta{tp}(convert.(tp, (4, 2))...)
        r = convert.(tp, rand(data_dist, nvars, n))
        r2 = convert.(tp, rand(data_dist, nvars, n))
        df = DataFrame(r', :auto)
        df2 = DataFrame(r2', :auto)

        # Flux Opt
        if mt <: CondPlanar
            nn = PlanarNN(nvars, tanh; cond=true)
        else
            nn = Chain(
                Dense(nvars*2, nvars, tanh),
            )
        end
        icnf = mt{tp}(nn, nvars; acceleration=cr)
        ufd = copy(icnf.p)
        model = CondICNFModel(icnf; n_epochs, batch_size, opt_app=FluxOptApp())
        mach = machine(model, (df, df2))
        @test !isnothing(fit!(mach))
        fd = MLJBase.fitted_params(mach).learned_parameters
        @test !isnothing(MLJBase.transform(mach, (df, df2)))
        @test fd != ufd

        # Optim Opt
        if mt <: CondPlanar
            nn = PlanarNN(nvars, tanh; cond=true)
        else
            nn = Chain(
                Dense(nvars*2, nvars, tanh),
            )
        end
        icnf = mt{tp}(nn, nvars; acceleration=cr)
        ufd = copy(icnf.p)
        model = CondICNFModel(icnf; n_epochs, batch_size, opt_app=OptimOptApp())
        mach = machine(model, (df, df2))
        @test !isnothing(fit!(mach))
        fd = MLJBase.fitted_params(mach).learned_parameters
        @test !isnothing(MLJBase.transform(mach, (df, df2)))
        @test fd != ufd

        # SciML Opt with Zygote
        if mt <: CondPlanar
            nn = PlanarNN(nvars, tanh; cond=true)
        else
            nn = Chain(
                Dense(nvars*2, nvars, tanh),
            )
        end
        icnf = mt{tp}(nn, nvars; acceleration=cr)
        ufd = copy(icnf.p)
        model = CondICNFModel(icnf; n_epochs, batch_size, opt_app=SciMLOptApp(), adtype=GalacticOptim.AutoZygote())
        mach = machine(model, (df, df2))
        @test !isnothing(fit!(mach))
        fd = MLJBase.fitted_params(mach).learned_parameters
        @test !isnothing(MLJBase.transform(mach, (df, df2)))
        @test fd != ufd

        # SciML Opt with ForwardDiff
        if mt <: CondPlanar
            nn = PlanarNN(nvars, tanh; cond=true)
        else
            nn = Chain(
                Dense(nvars*2, nvars, tanh),
            )
        end
        icnf = mt{tp}(nn, nvars; acceleration=cr)
        ufd = copy(icnf.p)
        model = CondICNFModel(icnf; n_epochs, batch_size, opt_app=SciMLOptApp(), adtype=GalacticOptim.AutoForwardDiff())
        mach = machine(model, (df, df2))
        @test !isnothing(fit!(mach))
        fd = MLJBase.fitted_params(mach).learned_parameters
        @test !isnothing(MLJBase.transform(mach, (df, df2)))
        @test fd != ufd
    end
end

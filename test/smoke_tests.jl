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
    opt_apps = ICNF.OptApp[FluxOptApp(), OptimOptApp(), SciMLOptApp()]
    go_oa = SciMLOptApp()
    go_ads = SciMLBase.AbstractADType[GalacticOptim.AutoZygote(), GalacticOptim.AutoForwardDiff()]
    go_mds = Any[ICNF.default_optimizer[FluxOptApp], ICNF.default_optimizer[OptimOptApp]]
    nvars_ = (1:2)
    n_epochs = 8
    batch_size = 8
    n = 8*4

    @testset "$mt | $(typeof(cr).name.name) | $tp | $nvars Vars" for
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
        @test !isnothing(loss_pn(icnf, r))
        @test !isnothing(loss_pln(icnf, r))
        @test !isnothing(loss_f(icnf, FluxOptApp())(r))
        @test !isnothing(loss_f(icnf, OptimOptApp(), [(r,), nothing])(icnf.p))
        @test !isnothing(loss_f(icnf, SciMLOptApp())(icnf.p, SciMLBase.NullParameters(), r))

        @test !isnothing(agg_loglikelihood(icnf, r))

        d = ICNFDist(icnf)

        @test !isnothing(logpdf(d, r))
        @test !isnothing(pdf(d, r))
        @test !isnothing(rand(d))
        @test !isnothing(rand(d, n))
    end
    @testset "Fitting | $mt | $(typeof(cr).name.name) | $tp | $nvars Vars" for
            mt in mts_fit,
            cr in crs,
            tp in tps_fit,
            nvars in nvars_
        data_dist = Beta{tp}(convert.(tp, (2, 4))...)
        r = convert.(tp, rand(data_dist, nvars, n))
        df = DataFrame(r', :auto)

        @testset "Using $(typeof(opt_app).name.name)" for
                opt_app in opt_apps
            if mt <: Planar
                nn = PlanarNN(nvars, tanh)
            else
                nn = Chain(
                    Dense(nvars, nvars, tanh),
                )
            end
            icnf = mt{tp}(nn, nvars; acceleration=cr)
            ufd = copy(icnf.p)
            model = ICNFModel(icnf; n_epochs, batch_size, opt_app)
            mach = machine(model, df)
            @test !isnothing(fit!(mach))
            fd = MLJBase.fitted_params(mach).learned_parameters
            @test !isnothing(MLJBase.transform(mach, df))
            @test fd != ufd
        end
        @testset "$(typeof(go_oa).name.name) | Using $(typeof(go_ad).name.name) & $(typeof(go_md).name.name)" for
                go_ad in go_ads,
                go_md in go_mds
            if mt <: Planar
                nn = PlanarNN(nvars, tanh)
            else
                nn = Chain(
                    Dense(nvars, nvars, tanh),
                )
            end
            icnf = mt{tp}(nn, nvars; acceleration=cr)
            ufd = copy(icnf.p)
            model = ICNFModel(icnf; n_epochs, batch_size, opt_app=go_oa, adtype=go_ad, optimizer=go_md)
            mach = machine(model, df)
            @test !isnothing(fit!(mach))
            fd = MLJBase.fitted_params(mach).learned_parameters
            @test !isnothing(MLJBase.transform(mach, df))
            @test fd != ufd
        end
    end
    @testset "$mt | $(typeof(cr).name.name) | $tp | $nvars Vars" for
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
        @test !isnothing(loss_pn(icnf, r, r2))
        @test !isnothing(loss_pln(icnf, r, r2))
        @test !isnothing(loss_f(icnf, FluxOptApp())(r, r2))
        @test !isnothing(loss_f(icnf, OptimOptApp(), [(r, r2), nothing])(icnf.p))
        @test !isnothing(loss_f(icnf, SciMLOptApp())(icnf.p, SciMLBase.NullParameters(), r, r2))

        @test !isnothing(agg_loglikelihood(icnf, r, r2))

        d = CondICNFDist(icnf, r2)

        @test !isnothing(logpdf(d, r))
        @test !isnothing(pdf(d, r))
        @test !isnothing(rand(d))
        @test !isnothing(rand(d, n))
    end
    @testset "Fitting | $mt | $(typeof(cr).name.name) | $tp | $nvars Vars" for
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

        @testset "Using $(typeof(opt_app).name.name)" for
                opt_app in opt_apps
            if mt <: CondPlanar
                nn = PlanarNN(nvars, tanh; cond=true)
            else
                nn = Chain(
                    Dense(nvars*2, nvars, tanh),
                )
            end
            icnf = mt{tp}(nn, nvars; acceleration=cr)
            ufd = copy(icnf.p)
            model = CondICNFModel(icnf; n_epochs, batch_size, opt_app)
            mach = machine(model, (df, df2))
            @test !isnothing(fit!(mach))
            fd = MLJBase.fitted_params(mach).learned_parameters
            @test !isnothing(MLJBase.transform(mach, (df, df2)))
            @test fd != ufd
        end
        @testset "$(typeof(go_oa).name.name) | Using $(typeof(go_ad).name.name) & $(typeof(go_md).name.name)" for
                go_ad in go_ads,
                go_md in go_mds
            if mt <: CondPlanar
                nn = PlanarNN(nvars, tanh; cond=true)
            else
                nn = Chain(
                    Dense(nvars*2, nvars, tanh),
                )
            end
            icnf = mt{tp}(nn, nvars; acceleration=cr)
            ufd = copy(icnf.p)
            model = CondICNFModel(icnf; n_epochs, batch_size, opt_app=go_oa, adtype=go_ad, optimizer=go_md)
            mach = machine(model, (df, df2))
            @test !isnothing(fit!(mach))
            fd = MLJBase.fitted_params(mach).learned_parameters
            @test !isnothing(MLJBase.transform(mach, (df, df2)))
            @test fd != ufd
        end
    end
end

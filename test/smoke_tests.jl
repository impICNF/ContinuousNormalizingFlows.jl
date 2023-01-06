@testset "Smoke Tests" begin
    mts = Type{<:ICNF.AbstractICNF}[RNODE, FFJORD, Planar]
    cmts = Type{<:ICNF.AbstractCondICNF}[CondRNODE, CondFFJORD, CondPlanar]
    ats = Type{<:AbstractArray}[Array]
    if has_cuda_gpu()
        push!(ats, CuArray)
    end
    tps = Type{<:AbstractFloat}[Float64, Float32, Float16]
    adb_list = AbstractDifferentiation.AbstractBackend[
        AbstractDifferentiation.ZygoteBackend(),
        AbstractDifferentiation.ReverseDiffBackend(),
        AbstractDifferentiation.ForwardDiffBackend(),
        AbstractDifferentiation.TrackerBackend(),
        AbstractDifferentiation.FiniteDifferencesBackend(),
    ]
    fd_m = FiniteDifferences.central_fdm(5, 1)
    opt_apps = ICNF.OptApp[FluxOptApp(), OptimOptApp(), SciMLOptApp()]
    go_oa = SciMLOptApp()
    go_ads = SciMLBase.AbstractADType[
        Optimization.AutoZygote(),
        Optimization.AutoReverseDiff(),
        Optimization.AutoForwardDiff(),
        Optimization.AutoTracker(),
        Optimization.AutoFiniteDiff(),
    ]
    go_mds = Any[
        ICNF.default_optimizer[FluxOptApp],
        ICNF.default_optimizer[OptimOptApp],
        ICNF.default_optimizer[SciMLOptApp],
    ]
    nvars_ = (1:2)
    n_epochs = 3
    batch_size = 3
    n_batch = 3
    n = n_batch * batch_size

    @testset "Call Tests" begin
        @testset "$at | $tp | $nvars Vars | $mt" for at in ats,
            tp in tps,
            nvars in nvars_,
            mt in mts

            data_dist = Beta{tp}(convert(Tuple{tp, tp}, (2, 4))...)
            r = convert(Matrix{tp}, rand(data_dist, nvars, n))

            if mt <: Planar
                nn = PlanarNN(nvars, tanh)
            else
                nn = Chain(Dense(nvars => nvars, tanh))
            end
            icnf = mt{tp, at}(nn, nvars)

            @test !isnothing(inference(icnf, TestMode(), r))
            @test !isnothing(inference(icnf, TrainMode(), r))
            @test !isnothing(generate(icnf, TestMode(), n))
            @test !isnothing(generate(icnf, TrainMode(), n))

            @test !isnothing(icnf(r))
            @test !isnothing(loss(icnf, r))
            @test !isnothing(loss_pn(icnf, r))
            @test !isnothing(loss_pln(icnf, r))
            @test !isnothing(loss_f(icnf, FluxOptApp())(icnf, r))
            @test !isnothing(loss_f(icnf, OptimOptApp(), [(r,), nothing])(icnf.p))
            @test !isnothing(
                loss_f(icnf, SciMLOptApp())(icnf.p, SciMLBase.NullParameters(), r),
            )

            @test !isnothing(agg_loglikelihood(icnf, r))

            diff_loss(x) = loss(icnf, r, x)

            @testset "Using $(typeof(adb).name.name)" for adb in adb_list
                @test_throws MethodError !isnothing(
                    AbstractDifferentiation.derivative(adb, diff_loss, icnf.p),
                )
                @test !isnothing(AbstractDifferentiation.gradient(adb, diff_loss, icnf.p))
                if adb isa AbstractDifferentiation.TrackerBackend
                    @test_throws MethodError !isnothing(
                        AbstractDifferentiation.jacobian(adb, diff_loss, icnf.p),
                    )
                else
                    @test !isnothing(
                        AbstractDifferentiation.jacobian(adb, diff_loss, icnf.p),
                    )
                end
                # @test !isnothing(AbstractDifferentiation.hessian(adb, diff_loss, icnf.p))
            end

            @test !isnothing(Zygote.gradient(diff_loss, icnf.p))
            @test !isnothing(Zygote.jacobian(diff_loss, icnf.p))
            @test !isnothing(Zygote.forwarddiff(diff_loss, icnf.p))
            # @test !isnothing(Zygote.diaghessian(diff_loss, icnf.p))
            # @test !isnothing(Zygote.hessian(diff_loss, icnf.p))
            # @test !isnothing(Zygote.hessian_reverse(diff_loss, icnf.p))

            @test !isnothing(ReverseDiff.gradient(diff_loss, icnf.p))
            @test_throws MethodError !isnothing(ReverseDiff.jacobian(diff_loss, icnf.p))
            # @test !isnothing(ReverseDiff.hessian(diff_loss, icnf.p))

            @test !isnothing(ForwardDiff.gradient(diff_loss, icnf.p))
            @test_throws DimensionMismatch !isnothing(
                ForwardDiff.jacobian(diff_loss, icnf.p),
            )
            # @test !isnothing(ForwardDiff.hessian(diff_loss, icnf.p))

            @test !isnothing(Tracker.gradient(diff_loss, icnf.p))
            @test_throws MethodError !isnothing(Tracker.jacobian(diff_loss, icnf.p))
            # @test !isnothing(Tracker.hessian(diff_loss, icnf.p))

            @test !isnothing(FiniteDifferences.grad(fd_m, diff_loss, icnf.p))
            @test !isnothing(FiniteDifferences.jacobian(fd_m, diff_loss, icnf.p))

            @test_throws MethodError !isnothing(
                FiniteDiff.finite_difference_derivative(diff_loss, icnf.p),
            )
            @test !isnothing(FiniteDiff.finite_difference_gradient(diff_loss, icnf.p))
            @test !isnothing(FiniteDiff.finite_difference_jacobian(diff_loss, icnf.p))
            # @test !isnothing(FiniteDiff.finite_difference_hessian(diff_loss, icnf.p))

            d = ICNFDist(icnf)

            @test !isnothing(logpdf(d, r))
            @test !isnothing(pdf(d, r))
            @test !isnothing(rand(d))
            @test !isnothing(rand(d, n))
        end
        @testset "$at | $tp | $nvars Vars | $mt" for at in ats,
            tp in tps,
            nvars in nvars_,
            mt in cmts

            data_dist = Beta{tp}(convert(Tuple{tp, tp}, (2, 4))...)
            data_dist2 = Beta{tp}(convert(Tuple{tp, tp}, (4, 2))...)
            r = convert(Matrix{tp}, rand(data_dist, nvars, n))
            r2 = convert(Matrix{tp}, rand(data_dist, nvars, n))

            if mt <: CondPlanar
                nn = PlanarNN(nvars, tanh; cond = true)
            else
                nn = Chain(Dense(2 * nvars => nvars, tanh))
            end
            icnf = mt{tp, at}(nn, nvars)

            @test !isnothing(inference(icnf, TestMode(), r, r2))
            @test !isnothing(inference(icnf, TrainMode(), r, r2))
            @test !isnothing(generate(icnf, TestMode(), r2, n))
            @test !isnothing(generate(icnf, TrainMode(), r2, n))

            @test !isnothing(icnf(r, r2))
            @test !isnothing(loss(icnf, r, r2))
            @test !isnothing(loss_pn(icnf, r, r2))
            @test !isnothing(loss_pln(icnf, r, r2))
            @test !isnothing(loss_f(icnf, FluxOptApp())(icnf, r, r2))
            @test !isnothing(loss_f(icnf, OptimOptApp(), [(r, r2), nothing])(icnf.p))
            @test !isnothing(
                loss_f(icnf, SciMLOptApp())(icnf.p, SciMLBase.NullParameters(), r, r2),
            )

            @test !isnothing(agg_loglikelihood(icnf, r, r2))

            diff_loss(x) = loss(icnf, r, r2, x)

            @testset "Using $(typeof(adb).name.name)" for adb in adb_list
                @test_throws MethodError !isnothing(
                    AbstractDifferentiation.derivative(adb, diff_loss, icnf.p),
                )
                @test !isnothing(AbstractDifferentiation.gradient(adb, diff_loss, icnf.p))
                if adb isa AbstractDifferentiation.TrackerBackend
                    @test_throws MethodError !isnothing(
                        AbstractDifferentiation.jacobian(adb, diff_loss, icnf.p),
                    )
                else
                    @test !isnothing(
                        AbstractDifferentiation.jacobian(adb, diff_loss, icnf.p),
                    )
                end
                # @test !isnothing(AbstractDifferentiation.hessian(adb, diff_loss, icnf.p))
            end

            @test !isnothing(Zygote.gradient(diff_loss, icnf.p))
            @test !isnothing(Zygote.jacobian(diff_loss, icnf.p))
            @test !isnothing(Zygote.forwarddiff(diff_loss, icnf.p))
            # @test !isnothing(Zygote.diaghessian(diff_loss, icnf.p))
            # @test !isnothing(Zygote.hessian(diff_loss, icnf.p))
            # @test !isnothing(Zygote.hessian_reverse(diff_loss, icnf.p))

            @test !isnothing(ReverseDiff.gradient(diff_loss, icnf.p))
            @test_throws MethodError !isnothing(ReverseDiff.jacobian(diff_loss, icnf.p))
            # @test !isnothing(ReverseDiff.hessian(diff_loss, icnf.p))

            @test !isnothing(ForwardDiff.gradient(diff_loss, icnf.p))
            @test_throws DimensionMismatch !isnothing(
                ForwardDiff.jacobian(diff_loss, icnf.p),
            )
            # @test !isnothing(ForwardDiff.hessian(diff_loss, icnf.p))

            @test !isnothing(Tracker.gradient(diff_loss, icnf.p))
            @test_throws MethodError !isnothing(Tracker.jacobian(diff_loss, icnf.p))
            # @test !isnothing(Tracker.hessian(diff_loss, icnf.p))

            @test !isnothing(FiniteDifferences.grad(fd_m, diff_loss, icnf.p))
            @test !isnothing(FiniteDifferences.jacobian(fd_m, diff_loss, icnf.p))

            @test_throws MethodError !isnothing(
                FiniteDiff.finite_difference_derivative(diff_loss, icnf.p),
            )
            @test !isnothing(FiniteDiff.finite_difference_gradient(diff_loss, icnf.p))
            @test !isnothing(FiniteDiff.finite_difference_jacobian(diff_loss, icnf.p))
            # @test !isnothing(FiniteDiff.finite_difference_hessian(diff_loss, icnf.p))

            d = CondICNFDist(icnf, r2)

            @test !isnothing(logpdf(d, r))
            @test !isnothing(pdf(d, r))
            @test !isnothing(rand(d))
            @test !isnothing(rand(d, n))
        end
    end
    @testset "Fit Tests" begin
        @testset "$at | $tp | $nvars Vars | $mt" for at in ats,
            tp in tps,
            nvars in nvars_,
            mt in mts

            data_dist = Beta{tp}(convert(Tuple{tp, tp}, (2, 4))...)
            r = convert(Matrix{tp}, rand(data_dist, nvars, n))
            df = DataFrame(transpose(r), :auto)

            @testset "Using $(typeof(opt_app).name.name)" for opt_app in opt_apps
                if mt <: Planar
                    nn = PlanarNN(nvars, tanh)
                else
                    nn = Chain(Dense(nvars => nvars, tanh))
                end
                icnf = mt{tp, at}(nn, nvars)
                model = ICNFModel(icnf; n_epochs, batch_size, opt_app)
                mach = machine(model, df)
                @test !isnothing(fit!(mach))
                @test !isnothing(MLJBase.transform(mach, df))
                @test !isnothing(MLJBase.fitted_params(mach))
            end
            @testset "$(typeof(go_oa).name.name) | Using $(typeof(go_ad).name.name) & $(typeof(go_md).name.name)" for go_ad in
                                                                                                                      go_ads,
                go_md in go_mds

                if mt <: Planar
                    nn = PlanarNN(nvars, tanh)
                else
                    nn = Chain(Dense(nvars => nvars, tanh))
                end
                icnf = mt{tp, at}(nn, nvars)
                model = ICNFModel(
                    icnf;
                    n_epochs,
                    batch_size,
                    opt_app = go_oa,
                    adtype = go_ad,
                    optimizer = go_md,
                )
                mach = machine(model, df)
                @test !isnothing(fit!(mach))
                @test !isnothing(MLJBase.transform(mach, df))
                @test !isnothing(MLJBase.fitted_params(mach))
            end
        end
        @testset "$at | $tp | $nvars Vars | $mt" for at in ats,
            tp in tps,
            nvars in nvars_,
            mt in cmts

            data_dist = Beta{tp}(convert(Tuple{tp, tp}, (2, 4))...)
            data_dist2 = Beta{tp}(convert(Tuple{tp, tp}, (4, 2))...)
            r = convert(Matrix{tp}, rand(data_dist, nvars, n))
            r2 = convert(Matrix{tp}, rand(data_dist, nvars, n))
            df = DataFrame(transpose(r), :auto)
            df2 = DataFrame(transpose(r2), :auto)

            @testset "Using $(typeof(opt_app).name.name)" for opt_app in opt_apps
                if mt <: CondPlanar
                    nn = PlanarNN(nvars, tanh; cond = true)
                else
                    nn = Chain(Dense(2 * nvars => nvars, tanh))
                end
                icnf = mt{tp, at}(nn, nvars)
                model = CondICNFModel(icnf; n_epochs, batch_size, opt_app)
                mach = machine(model, (df, df2))
                @test !isnothing(fit!(mach))
                @test !isnothing(MLJBase.transform(mach, (df, df2)))
                @test !isnothing(MLJBase.fitted_params(mach))
            end
            @testset "$(typeof(go_oa).name.name) | Using $(typeof(go_ad).name.name) & $(typeof(go_md).name.name)" for go_ad in
                                                                                                                      go_ads,
                go_md in go_mds

                if mt <: CondPlanar
                    nn = PlanarNN(nvars, tanh; cond = true)
                else
                    nn = Chain(Dense(2 * nvars => nvars, tanh))
                end
                icnf = mt{tp, at}(nn, nvars)
                model = CondICNFModel(
                    icnf;
                    n_epochs,
                    batch_size,
                    opt_app = go_oa,
                    adtype = go_ad,
                    optimizer = go_md,
                )
                mach = machine(model, (df, df2))
                @test !isnothing(fit!(mach))
                @test !isnothing(MLJBase.transform(mach, (df, df2)))
                @test !isnothing(MLJBase.fitted_params(mach))
            end
        end
    end
end

@testset "Call Tests" begin
    mts =
        SMALL ? Type{<:ICNF.AbstractICNF}[RNODE] :
        Type{<:ICNF.AbstractICNF}[RNODE, FFJORD, Planar]
    cmts =
        SMALL ? Type{<:ICNF.AbstractCondICNF}[CondRNODE] :
        Type{<:ICNF.AbstractCondICNF}[CondRNODE, CondFFJORD, CondPlanar]
    ats = Type{<:AbstractArray}[Array]
    if has_cuda_gpu() && !SMALL
        push!(ats, CuArray)
    end
    tps = Type{<:AbstractFloat}[Float64, Float32, Float16]
    nvars_ = (1:2)
    adb_list = AbstractDifferentiation.AbstractBackend[
        AbstractDifferentiation.ZygoteBackend(),
        AbstractDifferentiation.ReverseDiffBackend(),
        AbstractDifferentiation.ForwardDiffBackend(),
        AbstractDifferentiation.TrackerBackend(),
        AbstractDifferentiation.FiniteDifferencesBackend(),
    ]
    fd_m = FiniteDifferences.central_fdm(5, 1)

    @testset "$at | $tp | $nvars Vars | $mt" for at in ats,
        tp in tps,
        nvars in nvars_,
        mt in mts

        data_dist = Beta{tp}(convert(Tuple{tp, tp}, (2, 4))...)
        r = convert(Vector{tp}, rand(data_dist, nvars))
        r_arr = convert(Matrix{tp}, rand(data_dist, nvars, 2))

        if mt <: Planar
            nn = PlanarNN(nvars, tanh)
        else
            nn = Chain(Dense(nvars => nvars, tanh))
        end
        icnf = mt{tp, at}(nn, nvars)

        @test !isnothing(inference(icnf, TestMode(), r))
        @test !isnothing(inference(icnf, TestMode(), r_arr))
        @test !isnothing(inference(icnf, TrainMode(), r))
        @test !isnothing(inference(icnf, TrainMode(), r_arr))
        @test !isnothing(generate(icnf, TestMode()))
        @test !isnothing(generate(icnf, TestMode(), 2))
        @test !isnothing(generate(icnf, TrainMode()))
        @test !isnothing(generate(icnf, TrainMode(), 2))

        @test !isnothing(icnf(r))
        @test !isnothing(loss(icnf, r))
        @test !isnothing(loss(icnf, r_arr))
        @test !isnothing(loss_f(icnf, loss)(icnf.p, SciMLBase.NullParameters(), r_arr))

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
                @test !isnothing(AbstractDifferentiation.jacobian(adb, diff_loss, icnf.p))
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
        @test_throws DimensionMismatch !isnothing(ForwardDiff.jacobian(diff_loss, icnf.p))
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
        @test !isnothing(logpdf(d, r_arr))
        @test !isnothing(pdf(d, r))
        @test !isnothing(pdf(d, r_arr))
        @test !isnothing(rand(d))
        @test !isnothing(rand(d, 2))
    end
    @testset "$at | $tp | $nvars Vars | $mt" for at in ats,
        tp in tps,
        nvars in nvars_,
        mt in cmts

        data_dist = Beta{tp}(convert(Tuple{tp, tp}, (2, 4))...)
        data_dist2 = Beta{tp}(convert(Tuple{tp, tp}, (4, 2))...)
        r = convert(Vector{tp}, rand(data_dist, nvars))
        r_arr = convert(Matrix{tp}, rand(data_dist, nvars, 2))
        r2 = convert(Vector{tp}, rand(data_dist, nvars))
        r2_arr = convert(Matrix{tp}, rand(data_dist, nvars, 2))

        if mt <: CondPlanar
            nn = PlanarNN(nvars, tanh; cond = true)
        else
            nn = Chain(Dense(2 * nvars => nvars, tanh))
        end
        icnf = mt{tp, at}(nn, nvars)

        @test !isnothing(inference(icnf, TestMode(), r, r2))
        @test !isnothing(inference(icnf, TestMode(), r_arr, r2))
        @test !isnothing(inference(icnf, TrainMode(), r, r2))
        @test !isnothing(inference(icnf, TrainMode(), r_arr, r2))
        @test !isnothing(generate(icnf, TestMode(), r2))
        @test !isnothing(generate(icnf, TestMode(), r2, 2))
        @test !isnothing(generate(icnf, TrainMode(), r2))
        @test !isnothing(generate(icnf, TrainMode(), r2, 2))

        @test !isnothing(icnf(r, r2))
        @test !isnothing(loss(icnf, r, r2))
        @test !isnothing(loss(icnf, r_arr, r2_arr))
        @test !isnothing(
            loss_f(icnf, loss)(icnf.p, SciMLBase.NullParameters(), r_arr, r2_arr),
        )

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
                @test !isnothing(AbstractDifferentiation.jacobian(adb, diff_loss, icnf.p))
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
        @test_throws DimensionMismatch !isnothing(ForwardDiff.jacobian(diff_loss, icnf.p))
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
        @test !isnothing(logpdf(d, r_arr))
        @test !isnothing(pdf(d, r))
        @test !isnothing(pdf(d, r_arr))
        @test !isnothing(rand(d))
        @test !isnothing(rand(d, 2))
    end
end

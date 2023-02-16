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
    tps = Type{<:AbstractFloat}[Float32]
    cmodes =
        Type{<:ICNF.ComputeMode}[ZygoteMatrixMode, SDVecJacMatrixMode, SDJacVecMatrixMode]
    nvars_ = (1:2)
    adb_list = AbstractDifferentiation.AbstractBackend[
        AbstractDifferentiation.ZygoteBackend(),
        AbstractDifferentiation.ReverseDiffBackend(),
        AbstractDifferentiation.ForwardDiffBackend(),
        AbstractDifferentiation.TrackerBackend(),
        AbstractDifferentiation.FiniteDifferencesBackend(),
    ]
    fd_m = FiniteDifferences.central_fdm(5, 1)
    rng = Random.default_rng()

    @testset "$at | $tp | $(typeof(adb_u).name.name) | $nvars Vars | $mt" for at in ats,
        tp in tps,
        adb_u in adb_list,
        nvars in nvars_,
        mt in mts

        adb_u isa AbstractDifferentiation.FiniteDifferencesBackend && continue
        adb_u isa AbstractDifferentiation.ReverseDiffBackend && continue
        adb_u isa AbstractDifferentiation.TrackerBackend && mt <: Planar && continue

        data_dist = Beta{tp}(convert(Tuple{tp, tp}, (2, 4))...)
        r = convert(at{tp}, rand(data_dist, nvars))
        r_arr = convert(at{tp}, rand(data_dist, nvars, 2))

        if mt <: Planar
            nn = PlanarLayer(nvars, tanh)
        else
            nn = Dense(nvars => nvars, tanh)
        end
        icnf = construct(
            mt,
            nn,
            nvars;
            data_type = tp,
            array_type = at,
            differentiation_backend = adb_u,
        )
        ps, st = Lux.setup(rng, icnf)
        ps = ComponentArray(map(at{tp}, ps))

        @test !isnothing(inference(icnf, TestMode(), r, ps, st))
        @test !isnothing(inference(icnf, TrainMode(), r, ps, st))
        @test !isnothing(generate(icnf, TestMode(), ps, st))
        @test !isnothing(generate(icnf, TrainMode(), ps, st))

        @test !isnothing(loss(icnf, r, ps, st))

        diff_loss(x) = loss(icnf, r, x, st)

        @testset "Using $(typeof(adb).name.name) For Loss" for adb in adb_list
            @test_throws MethodError !isnothing(
                AbstractDifferentiation.derivative(adb, diff_loss, ps),
            )
            @test !isnothing(AbstractDifferentiation.gradient(adb, diff_loss, ps))
            if adb isa AbstractDifferentiation.TrackerBackend
                @test_throws MethodError !isnothing(
                    AbstractDifferentiation.jacobian(adb, diff_loss, ps),
                )
            else
                @test !isnothing(AbstractDifferentiation.jacobian(adb, diff_loss, ps))
            end
            # @test !isnothing(AbstractDifferentiation.hessian(adb, diff_loss, ps))
        end

        @test !isnothing(Zygote.gradient(diff_loss, ps))
        @test !isnothing(Zygote.jacobian(diff_loss, ps))
        @test !isnothing(Zygote.forwarddiff(diff_loss, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss, ps))
        # @test !isnothing(Zygote.hessian(diff_loss, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss, ps))

        @test !isnothing(ReverseDiff.gradient(diff_loss, ps))
        @test_throws MethodError !isnothing(ReverseDiff.jacobian(diff_loss, ps))
        # @test !isnothing(ReverseDiff.hessian(diff_loss, ps))

        @test !isnothing(ForwardDiff.gradient(diff_loss, ps))
        @test_throws DimensionMismatch !isnothing(ForwardDiff.jacobian(diff_loss, ps))
        # @test !isnothing(ForwardDiff.hessian(diff_loss, ps))

        @test !isnothing(Tracker.gradient(diff_loss, ps))
        @test_throws MethodError !isnothing(Tracker.jacobian(diff_loss, ps))
        # @test !isnothing(Tracker.hessian(diff_loss, ps))

        @test !isnothing(FiniteDifferences.grad(fd_m, diff_loss, ps))
        @test !isnothing(FiniteDifferences.jacobian(fd_m, diff_loss, ps))

        @test_throws MethodError !isnothing(
            FiniteDiff.finite_difference_derivative(diff_loss, ps),
        )
        @test !isnothing(FiniteDiff.finite_difference_gradient(diff_loss, ps))
        @test !isnothing(FiniteDiff.finite_difference_jacobian(diff_loss, ps))
        # @test !isnothing(FiniteDiff.finite_difference_hessian(diff_loss, ps))

        d = ICNFDist(icnf, ps, st)

        @test !isnothing(logpdf(d, r))
        @test !isnothing(logpdf(d, r_arr))
        @test !isnothing(pdf(d, r))
        @test !isnothing(pdf(d, r_arr))
        @test !isnothing(rand(d))
        @test !isnothing(rand(d, 2))
    end
    @testset "$at | $tp | $cmode | $nvars Vars | $mt" for at in ats,
        tp in tps,
        cmode in cmodes,
        nvars in nvars_,
        mt in mts

        cmode <: SDJacVecMatrixMode && continue

        data_dist = Beta{tp}(convert(Tuple{tp, tp}, (2, 4))...)
        r = convert(at{tp}, rand(data_dist, nvars))
        r_arr = convert(at{tp}, rand(data_dist, nvars, 2))

        if mt <: Planar
            nn = PlanarLayer(nvars, tanh)
        else
            nn = Dense(nvars => nvars, tanh)
        end
        icnf =
            construct(mt, nn, nvars; data_type = tp, array_type = at, compute_mode = cmode)
        ps, st = Lux.setup(rng, icnf)
        ps = ComponentArray(map(at{tp}, ps))

        @test !isnothing(inference(icnf, TestMode(), r_arr, ps, st))
        @test !isnothing(inference(icnf, TrainMode(), r_arr, ps, st))
        @test !isnothing(generate(icnf, TestMode(), ps, st, 2))
        @test !isnothing(generate(icnf, TrainMode(), ps, st, 2))

        @test !isnothing(loss(icnf, r_arr, ps, st))

        diff_loss(x) = loss(icnf, r_arr, x, st)

        @testset "Using $(typeof(adb).name.name) For Loss" for adb in adb_list
            @test_throws MethodError !isnothing(
                AbstractDifferentiation.derivative(adb, diff_loss, ps),
            )
            @test !isnothing(AbstractDifferentiation.gradient(adb, diff_loss, ps))
            if adb isa AbstractDifferentiation.TrackerBackend
                @test_throws MethodError !isnothing(
                    AbstractDifferentiation.jacobian(adb, diff_loss, ps),
                )
            else
                @test !isnothing(AbstractDifferentiation.jacobian(adb, diff_loss, ps))
            end
            # @test !isnothing(AbstractDifferentiation.hessian(adb, diff_loss, ps))
        end

        @test !isnothing(Zygote.gradient(diff_loss, ps))
        @test !isnothing(Zygote.jacobian(diff_loss, ps))
        @test !isnothing(Zygote.forwarddiff(diff_loss, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss, ps))
        # @test !isnothing(Zygote.hessian(diff_loss, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss, ps))

        @test !isnothing(ReverseDiff.gradient(diff_loss, ps))
        @test_throws MethodError !isnothing(ReverseDiff.jacobian(diff_loss, ps))
        # @test !isnothing(ReverseDiff.hessian(diff_loss, ps))

        @test !isnothing(ForwardDiff.gradient(diff_loss, ps))
        @test_throws DimensionMismatch !isnothing(ForwardDiff.jacobian(diff_loss, ps))
        # @test !isnothing(ForwardDiff.hessian(diff_loss, ps))

        @test !isnothing(Tracker.gradient(diff_loss, ps))
        @test_throws MethodError !isnothing(Tracker.jacobian(diff_loss, ps))
        # @test !isnothing(Tracker.hessian(diff_loss, ps))

        @test !isnothing(FiniteDifferences.grad(fd_m, diff_loss, ps))
        @test !isnothing(FiniteDifferences.jacobian(fd_m, diff_loss, ps))

        @test_throws MethodError !isnothing(
            FiniteDiff.finite_difference_derivative(diff_loss, ps),
        )
        @test !isnothing(FiniteDiff.finite_difference_gradient(diff_loss, ps))
        @test !isnothing(FiniteDiff.finite_difference_jacobian(diff_loss, ps))
        # @test !isnothing(FiniteDiff.finite_difference_hessian(diff_loss, ps))

        d = ICNFDist(icnf, ps, st)

        @test !isnothing(logpdf(d, r))
        @test !isnothing(logpdf(d, r_arr))
        @test !isnothing(pdf(d, r))
        @test !isnothing(pdf(d, r_arr))
        @test !isnothing(rand(d))
        @test !isnothing(rand(d, 2))
    end
    @testset "$at | $tp | $(typeof(adb_u).name.name) | $nvars Vars | $mt" for at in ats,
        tp in tps,
        adb_u in adb_list,
        nvars in nvars_,
        mt in cmts

        adb_u isa AbstractDifferentiation.FiniteDifferencesBackend && continue
        adb_u isa AbstractDifferentiation.ReverseDiffBackend && continue
        adb_u isa AbstractDifferentiation.TrackerBackend && continue
        adb_u isa AbstractDifferentiation.TrackerBackend && mt <: CondPlanar && continue

        data_dist = Beta{tp}(convert(Tuple{tp, tp}, (2, 4))...)
        data_dist2 = Beta{tp}(convert(Tuple{tp, tp}, (4, 2))...)
        r = convert(at{tp}, rand(data_dist, nvars))
        r_arr = convert(at{tp}, rand(data_dist, nvars, 2))
        r2 = convert(at{tp}, rand(data_dist, nvars))
        r2_arr = convert(at{tp}, rand(data_dist, nvars, 2))

        if mt <: CondPlanar
            nn = PlanarLayer(nvars, tanh; cond = true)
        else
            nn = Dense(2 * nvars => nvars, tanh)
        end
        icnf = construct(
            mt,
            nn,
            nvars;
            data_type = tp,
            array_type = at,
            differentiation_backend = adb_u,
        )
        ps, st = Lux.setup(rng, icnf)
        ps = ComponentArray(map(at{tp}, ps))

        @test !isnothing(inference(icnf, TestMode(), r, r2, ps, st))
        @test !isnothing(inference(icnf, TrainMode(), r, r2, ps, st))
        @test !isnothing(generate(icnf, TestMode(), r2, ps, st))
        @test !isnothing(generate(icnf, TrainMode(), r2, ps, st))

        @test !isnothing(loss(icnf, r, r2, ps, st))

        diff_loss(x) = loss(icnf, r, r2, x, st)

        @testset "Using $(typeof(adb).name.name) For Loss" for adb in adb_list
            @test_throws MethodError !isnothing(
                AbstractDifferentiation.derivative(adb, diff_loss, ps),
            )
            @test !isnothing(AbstractDifferentiation.gradient(adb, diff_loss, ps))
            if adb isa AbstractDifferentiation.TrackerBackend
                @test_throws MethodError !isnothing(
                    AbstractDifferentiation.jacobian(adb, diff_loss, ps),
                )
            else
                @test !isnothing(AbstractDifferentiation.jacobian(adb, diff_loss, ps))
            end
            # @test !isnothing(AbstractDifferentiation.hessian(adb, diff_loss, ps))
        end

        @test !isnothing(Zygote.gradient(diff_loss, ps))
        @test !isnothing(Zygote.jacobian(diff_loss, ps))
        @test !isnothing(Zygote.forwarddiff(diff_loss, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss, ps))
        # @test !isnothing(Zygote.hessian(diff_loss, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss, ps))

        @test !isnothing(ReverseDiff.gradient(diff_loss, ps))
        @test_throws MethodError !isnothing(ReverseDiff.jacobian(diff_loss, ps))
        # @test !isnothing(ReverseDiff.hessian(diff_loss, ps))

        @test !isnothing(ForwardDiff.gradient(diff_loss, ps))
        @test_throws DimensionMismatch !isnothing(ForwardDiff.jacobian(diff_loss, ps))
        # @test !isnothing(ForwardDiff.hessian(diff_loss, ps))

        @test !isnothing(Tracker.gradient(diff_loss, ps))
        @test_throws MethodError !isnothing(Tracker.jacobian(diff_loss, ps))
        # @test !isnothing(Tracker.hessian(diff_loss, ps))

        @test !isnothing(FiniteDifferences.grad(fd_m, diff_loss, ps))
        @test !isnothing(FiniteDifferences.jacobian(fd_m, diff_loss, ps))

        @test_throws MethodError !isnothing(
            FiniteDiff.finite_difference_derivative(diff_loss, ps),
        )
        @test !isnothing(FiniteDiff.finite_difference_gradient(diff_loss, ps))
        @test !isnothing(FiniteDiff.finite_difference_jacobian(diff_loss, ps))
        # @test !isnothing(FiniteDiff.finite_difference_hessian(diff_loss, ps))

        d = CondICNFDist(icnf, r2, ps, st)

        @test !isnothing(logpdf(d, r))
        @test !isnothing(logpdf(d, r_arr))
        @test !isnothing(pdf(d, r))
        @test !isnothing(pdf(d, r_arr))
        @test !isnothing(rand(d))
        @test !isnothing(rand(d, 2))
    end
    @testset "$at | $tp | $cmode | $nvars Vars | $mt" for at in ats,
        tp in tps,
        cmode in cmodes,
        nvars in nvars_,
        mt in cmts

        cmode <: SDJacVecMatrixMode && continue

        data_dist = Beta{tp}(convert(Tuple{tp, tp}, (2, 4))...)
        data_dist2 = Beta{tp}(convert(Tuple{tp, tp}, (4, 2))...)
        r = convert(at{tp}, rand(data_dist, nvars))
        r_arr = convert(at{tp}, rand(data_dist, nvars, 2))
        r2 = convert(at{tp}, rand(data_dist, nvars))
        r2_arr = convert(at{tp}, rand(data_dist, nvars, 2))

        if mt <: CondPlanar
            nn = PlanarLayer(nvars, tanh; cond = true)
        else
            nn = Dense(2 * nvars => nvars, tanh)
        end
        icnf =
            construct(mt, nn, nvars; data_type = tp, array_type = at, compute_mode = cmode)
        ps, st = Lux.setup(rng, icnf)
        ps = ComponentArray(map(at{tp}, ps))

        @test !isnothing(inference(icnf, TestMode(), r_arr, r2_arr, ps, st))
        @test !isnothing(inference(icnf, TrainMode(), r_arr, r2_arr, ps, st))
        @test !isnothing(generate(icnf, TestMode(), r2_arr, ps, st, 2))
        @test !isnothing(generate(icnf, TrainMode(), r2_arr, ps, st, 2))

        @test !isnothing(loss(icnf, r_arr, r2_arr, ps, st))

        diff_loss(x) = loss(icnf, r_arr, r2_arr, x, st)

        @testset "Using $(typeof(adb).name.name) For Loss" for adb in adb_list
            @test_throws MethodError !isnothing(
                AbstractDifferentiation.derivative(adb, diff_loss, ps),
            )
            @test !isnothing(AbstractDifferentiation.gradient(adb, diff_loss, ps))
            if adb isa AbstractDifferentiation.TrackerBackend
                @test_throws MethodError !isnothing(
                    AbstractDifferentiation.jacobian(adb, diff_loss, ps),
                )
            else
                @test !isnothing(AbstractDifferentiation.jacobian(adb, diff_loss, ps))
            end
            # @test !isnothing(AbstractDifferentiation.hessian(adb, diff_loss, ps))
        end

        @test !isnothing(Zygote.gradient(diff_loss, ps))
        @test !isnothing(Zygote.jacobian(diff_loss, ps))
        @test !isnothing(Zygote.forwarddiff(diff_loss, ps))
        # @test !isnothing(Zygote.diaghessian(diff_loss, ps))
        # @test !isnothing(Zygote.hessian(diff_loss, ps))
        # @test !isnothing(Zygote.hessian_reverse(diff_loss, ps))

        @test !isnothing(ReverseDiff.gradient(diff_loss, ps))
        @test_throws MethodError !isnothing(ReverseDiff.jacobian(diff_loss, ps))
        # @test !isnothing(ReverseDiff.hessian(diff_loss, ps))

        @test !isnothing(ForwardDiff.gradient(diff_loss, ps))
        @test_throws DimensionMismatch !isnothing(ForwardDiff.jacobian(diff_loss, ps))
        # @test !isnothing(ForwardDiff.hessian(diff_loss, ps))

        @test !isnothing(Tracker.gradient(diff_loss, ps))
        @test_throws MethodError !isnothing(Tracker.jacobian(diff_loss, ps))
        # @test !isnothing(Tracker.hessian(diff_loss, ps))

        @test !isnothing(FiniteDifferences.grad(fd_m, diff_loss, ps))
        @test !isnothing(FiniteDifferences.jacobian(fd_m, diff_loss, ps))

        @test_throws MethodError !isnothing(
            FiniteDiff.finite_difference_derivative(diff_loss, ps),
        )
        @test !isnothing(FiniteDiff.finite_difference_gradient(diff_loss, ps))
        @test !isnothing(FiniteDiff.finite_difference_jacobian(diff_loss, ps))
        # @test !isnothing(FiniteDiff.finite_difference_hessian(diff_loss, ps))

        d = CondICNFDist(icnf, r2_arr, ps, st)

        @test !isnothing(logpdf(d, r))
        @test !isnothing(logpdf(d, r_arr))
        @test !isnothing(pdf(d, r))
        @test !isnothing(pdf(d, r_arr))
        @test !isnothing(rand(d))
        @test !isnothing(rand(d, 2))
    end
end

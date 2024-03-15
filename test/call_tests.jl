@testset "Call Tests" begin
    mts = if GROUP == "RNODE"
        Type{<:ContinuousNormalizingFlows.AbstractFlows}[RNODE]
    elseif GROUP == "FFJORD"
        Type{<:ContinuousNormalizingFlows.AbstractFlows}[FFJORD]
    elseif GROUP == "Planar"
        Type{<:ContinuousNormalizingFlows.AbstractFlows}[Planar]
    elseif GROUP == "CondRNODE"
        Type{<:ContinuousNormalizingFlows.AbstractFlows}[CondRNODE]
    elseif GROUP == "CondFFJORD"
        Type{<:ContinuousNormalizingFlows.AbstractFlows}[CondFFJORD]
    elseif GROUP == "CondPlanar"
        Type{<:ContinuousNormalizingFlows.AbstractFlows}[CondPlanar]
    else
        Type{<:ContinuousNormalizingFlows.AbstractFlows}[
            RNODE,
            FFJORD,
            Planar,
            CondRNODE,
            CondFFJORD,
            CondPlanar,
        ]
    end
    omodes = ContinuousNormalizingFlows.Mode[TrainMode(), TestMode()]
    ndata_ = Int[4]
    nvars_ = Int[2]
    aug_steers = Bool[false, true]
    inplaces = Bool[false, true]
    adb_list = AbstractDifferentiation.AbstractBackend[
        AbstractDifferentiation.ZygoteBackend(),
        AbstractDifferentiation.ReverseDiffBackend(),
        AbstractDifferentiation.ForwardDiffBackend(),
    ]
    compute_modes = Type{<:ContinuousNormalizingFlows.ComputeMode}[
        ADVecJacVectorMode,
        ADJacVecVectorMode,
        ZygoteVectorMode,
        SDVecJacMatrixMode,
        SDJacVecMatrixMode,
        ZygoteMatrixMode,
    ]
    data_types = Type{<:AbstractFloat}[Float32]
    resources = ComputationalResources.AbstractResource[ComputationalResources.CPU1()]
    if CUDA.has_cuda_gpu() && USE_GPU
        push!(resources, ComputationalResources.CUDALibs())
        gdev = Lux.gpu_device()
    end

    @testset "$resource | $data_type | $compute_mode | inplace = $inplace | aug & steer = $aug_steer | nvars = $nvars | $omode | $mt" for resource in
                                                                                                                                          resources,
        data_type in data_types,
        compute_mode in compute_modes,
        inplace in inplaces,
        aug_steer in aug_steers,
        nvars in nvars_,
        ndata in ndata_,
        omode in omodes,
        mt in mts

        data_dist =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
        data_dist2 =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (4, 2))...)
        if compute_mode <: ContinuousNormalizingFlows.VectorMode
            r = convert.(data_type, rand(data_dist, nvars))
            r2 = convert.(data_type, rand(data_dist2, nvars))
        elseif compute_mode <: ContinuousNormalizingFlows.MatrixMode
            r = convert.(data_type, rand(data_dist, nvars, ndata))
            r2 = convert.(data_type, rand(data_dist2, nvars, ndata))
        end

        nn = ifelse(
            mt <: Union{CondRNODE, CondFFJORD, CondPlanar},
            ifelse(
                mt <: CondPlanar,
                ifelse(
                    aug_steer,
                    PlanarLayer(nvars * 2, tanh; n_cond = nvars),
                    PlanarLayer(nvars, tanh; n_cond = nvars),
                ),
                ifelse(
                    aug_steer,
                    Lux.Dense(nvars * 3 => nvars * 2, tanh),
                    Lux.Dense(nvars * 2 => nvars, tanh),
                ),
            ),
            ifelse(
                mt <: Planar,
                ifelse(aug_steer, PlanarLayer(nvars * 2, tanh), PlanarLayer(nvars, tanh)),
                ifelse(
                    aug_steer,
                    Lux.Dense(nvars * 2 => nvars * 2, tanh),
                    Lux.Dense(nvars => nvars, tanh),
                ),
            ),
        )
        icnf = ifelse(
            aug_steer,
            construct(
                mt,
                nn,
                nvars,
                nvars;
                data_type,
                compute_mode,
                inplace,
                resource,
                steer_rate = convert(data_type, 1e-1),
                λ₃ = convert(data_type, 1e-2),
            ),
            construct(
                mt,
                nn,
                nvars;
                data_type,
                compute_mode,
                inplace,
                resource,
                λ₃ = convert(data_type, 1e-2),
            ),
        )
        ps, st = Lux.setup(icnf.rng, icnf)
        ps = ComponentArrays.ComponentArray(ps)
        if resource isa ComputationalResources.CUDALibs
            r = gdev(r)
            r2 = gdev(r2)
            ps = gdev(ps)
            st = gdev(st)
        end

        if mt <: Union{CondRNODE, CondFFJORD, CondPlanar}
            @test !isnothing(inference(icnf, omode, r, r2, ps, st))
            if compute_mode <: ContinuousNormalizingFlows.MatrixMode
                @test !isnothing(generate(icnf, omode, r2, ps, st, ndata))
            else
                @test !isnothing(generate(icnf, omode, r2, ps, st))
            end

            @test !isnothing(loss(icnf, omode, r, r2, ps, st))

            diff_loss = x -> loss(icnf, omode, r, r2, x, st)
            diff2_loss = x -> loss(icnf, omode, x, r2, ps, st)
        else
            @test !isnothing(inference(icnf, omode, r, ps, st))
            if compute_mode <: ContinuousNormalizingFlows.MatrixMode
                @test !isnothing(generate(icnf, omode, ps, st, ndata))
            else
                @test !isnothing(generate(icnf, omode, ps, st))
            end

            @test !isnothing(loss(icnf, omode, r, ps, st))

            diff_loss = x -> loss(icnf, omode, r, x, st)
            diff2_loss = x -> loss(icnf, omode, x, ps, st)
        end

        if mt <: Union{CondRNODE, CondFFJORD, CondPlanar}
            d = CondICNFDist(icnf, omode, r2, ps, st)
        else
            d = ICNFDist(icnf, omode, ps, st)
        end

        @test !isnothing(Distributions.logpdf(d, r))
        @test !isnothing(Distributions.pdf(d, r))
        @test !isnothing(rand(d))
        @test !isnothing(rand(d, ndata))

        if (GROUP != "All") && (compute_mode <: SDJacVecMatrixMode || inplace)
            continue
        end

        @testset "$(typeof(adb).name.name) / Loss / ps" for adb in adb_list
            @test_throws MethodError !isnothing(
                AbstractDifferentiation.derivative(adb, diff_loss, ps),
            )
            @test !isnothing(AbstractDifferentiation.gradient(adb, diff_loss, ps))
            @test !isnothing(AbstractDifferentiation.jacobian(adb, diff_loss, ps))
            @test !isnothing(AbstractDifferentiation.hessian(adb, diff_loss, ps)) skip =
                (GROUP != "All")
        end

        @test !isnothing(Zygote.gradient(diff_loss, ps))
        @test !isnothing(Zygote.jacobian(diff_loss, ps))
        @test !isnothing(Zygote.diaghessian(diff_loss, ps)) skip = (GROUP != "All")
        @test !isnothing(Zygote.hessian(diff_loss, ps)) skip = (GROUP != "All")
        @test !isnothing(Zygote.hessian_reverse(diff_loss, ps)) skip = (GROUP != "All")
        diff_loss2 = x -> Zygote.checkpointed(diff_loss, x)
        @test !isnothing(Zygote.gradient(diff_loss2, ps))
        @test !isnothing(Zygote.jacobian(diff_loss2, ps))
        @test !isnothing(Zygote.diaghessian(diff_loss2, ps)) skip = (GROUP != "All")
        @test !isnothing(Zygote.hessian(diff_loss2, ps)) skip = (GROUP != "All")
        @test !isnothing(Zygote.hessian_reverse(diff_loss2, ps)) skip = (GROUP != "All")
        diff_loss3 = x -> Zygote.forwarddiff(diff_loss, x)
        @test !isnothing(Zygote.gradient(diff_loss3, ps)) broken = (GROUP != "All")
        @test !isnothing(Zygote.jacobian(diff_loss3, ps)) broken = (GROUP != "All")
        @test !isnothing(Zygote.diaghessian(diff_loss3, ps)) skip = (GROUP != "All")
        @test !isnothing(Zygote.hessian(diff_loss3, ps)) skip = (GROUP != "All")
        @test !isnothing(Zygote.hessian_reverse(diff_loss3, ps)) skip = (GROUP != "All")
        diff_loss4 = x -> Zygote.forwarddiff(diff_loss2, x)
        @test !isnothing(Zygote.gradient(diff_loss4, ps)) broken = (GROUP != "All")
        @test !isnothing(Zygote.jacobian(diff_loss4, ps)) broken = (GROUP != "All")
        @test !isnothing(Zygote.diaghessian(diff_loss4, ps)) skip = (GROUP != "All")
        @test !isnothing(Zygote.hessian(diff_loss4, ps)) skip = (GROUP != "All")
        @test !isnothing(Zygote.hessian_reverse(diff_loss4, ps)) skip = (GROUP != "All")

        @test !isnothing(ReverseDiff.gradient(diff_loss, ps))
        @test_throws MethodError !isnothing(ReverseDiff.jacobian(diff_loss, ps))
        @test !isnothing(ReverseDiff.hessian(diff_loss, ps)) skip = (GROUP != "All")

        @test !isnothing(ForwardDiff.gradient(diff_loss, ps))
        @test_throws DimensionMismatch !isnothing(ForwardDiff.jacobian(diff_loss, ps))
        @test !isnothing(ForwardDiff.hessian(diff_loss, ps)) skip = (GROUP != "All")

        @testset "$(typeof(adb).name.name) / Loss / x" for adb in adb_list
            @test_throws MethodError !isnothing(
                AbstractDifferentiation.derivative(adb, diff2_loss, r),
            )
            @test !isnothing(AbstractDifferentiation.gradient(adb, diff2_loss, r)) broken =
                (GROUP != "All") &&
                compute_mode <: Union{ZygoteMatrixMode, SDVecJacMatrixMode} &&
                adb isa AbstractDifferentiation.ReverseDiffBackend &&
                VERSION >= v"1.10"
            @test !isnothing(AbstractDifferentiation.jacobian(adb, diff2_loss, r)) broken =
                (GROUP != "All") &&
                compute_mode <: Union{ZygoteMatrixMode, SDVecJacMatrixMode} &&
                adb isa AbstractDifferentiation.ReverseDiffBackend &&
                VERSION >= v"1.10"
            @test !isnothing(AbstractDifferentiation.hessian(adb, diff2_loss, r)) skip =
                (GROUP != "All")
        end

        @test !isnothing(Zygote.gradient(diff2_loss, r))
        @test !isnothing(Zygote.jacobian(diff2_loss, r))
        @test !isnothing(Zygote.diaghessian(diff2_loss, r)) skip = (GROUP != "All")
        @test !isnothing(Zygote.hessian(diff2_loss, r)) skip = (GROUP != "All")
        @test !isnothing(Zygote.hessian_reverse(diff2_loss, r)) skip = (GROUP != "All")
        diff2_loss2 = x -> Zygote.checkpointed(diff2_loss, x)
        @test !isnothing(Zygote.gradient(diff2_loss2, r))
        @test !isnothing(Zygote.jacobian(diff2_loss2, r))
        @test !isnothing(Zygote.diaghessian(diff2_loss2, r)) skip = (GROUP != "All")
        @test !isnothing(Zygote.hessian(diff2_loss2, r)) skip = (GROUP != "All")
        @test !isnothing(Zygote.hessian_reverse(diff2_loss2, r)) skip = (GROUP != "All")
        diff2_loss3 = x -> Zygote.forwarddiff(diff2_loss, x)
        @test !isnothing(Zygote.gradient(diff2_loss3, r))
        @test !isnothing(Zygote.jacobian(diff2_loss3, r))
        @test !isnothing(Zygote.diaghessian(diff2_loss3, r)) skip = (GROUP != "All")
        @test !isnothing(Zygote.hessian(diff2_loss3, r)) skip = (GROUP != "All")
        @test !isnothing(Zygote.hessian_reverse(diff2_loss3, r)) skip = (GROUP != "All")
        diff2_loss4 = x -> Zygote.forwarddiff(diff2_loss2, x)
        @test !isnothing(Zygote.gradient(diff2_loss4, r))
        @test !isnothing(Zygote.jacobian(diff2_loss4, r))
        @test !isnothing(Zygote.diaghessian(diff2_loss4, r)) skip = (GROUP != "All")
        @test !isnothing(Zygote.hessian(diff2_loss4, r)) skip = (GROUP != "All")
        @test !isnothing(Zygote.hessian_reverse(diff2_loss4, r)) skip = (GROUP != "All")

        @test !isnothing(ReverseDiff.gradient(diff2_loss, r)) broken =
            (GROUP != "All") &&
            compute_mode <: Union{ZygoteMatrixMode, SDVecJacMatrixMode} &&
            VERSION >= v"1.10"
        @test_throws MethodError !isnothing(ReverseDiff.jacobian(diff2_loss, r))
        @test !isnothing(ReverseDiff.hessian(diff2_loss, r)) skip = (GROUP != "All")

        @test !isnothing(ForwardDiff.gradient(diff2_loss, r))
        @test_throws DimensionMismatch !isnothing(ForwardDiff.jacobian(diff2_loss, r))
        @test !isnothing(ForwardDiff.hessian(diff2_loss, r)) skip = (GROUP != "All")
    end
end

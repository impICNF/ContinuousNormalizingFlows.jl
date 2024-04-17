@testset "Call Tests" begin
    mts = if GROUP == "RNODE"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[RNODE]
    elseif GROUP == "FFJORD"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[FFJORD]
    elseif GROUP == "Planar"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[Planar]
    elseif GROUP == "CondRNODE"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[CondRNODE]
    elseif GROUP == "CondFFJORD"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[CondFFJORD]
    elseif GROUP == "CondPlanar"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[CondPlanar]
    else
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[
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
    adtypes = ADTypes.AbstractADType[
        ADTypes.AutoZygote(),
        ADTypes.AutoReverseDiff(),
        ADTypes.AutoForwardDiff(),
    ]
    compute_modes = Type{<:ContinuousNormalizingFlows.ComputeMode}[
        ADVecJacVectorMode,
        ADJacVecVectorMode,
        DIVecJacVectorMode,
        DIJacVecVectorMode,
        DIVecJacMatrixMode,
        DIJacVecMatrixMode,
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
                    Lux.Chain(PlanarLayer(nvars * 2, tanh; n_cond = nvars)),
                    Lux.Chain(PlanarLayer(nvars, tanh; n_cond = nvars)),
                ),
                ifelse(
                    aug_steer,
                    Lux.Chain(Lux.Dense(nvars * 3 => nvars * 2, tanh)),
                    Lux.Chain(Lux.Dense(nvars * 2 => nvars, tanh)),
                ),
            ),
            ifelse(
                mt <: Planar,
                ifelse(
                    aug_steer,
                    Lux.Chain(PlanarLayer(nvars * 2, tanh)),
                    Lux.Chain(PlanarLayer(nvars, tanh)),
                ),
                ifelse(
                    aug_steer,
                    Lux.Chain(Lux.Dense(nvars * 2 => nvars * 2, tanh)),
                    Lux.Chain(Lux.Dense(nvars => nvars, tanh)),
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
            construct(mt, nn, nvars; data_type, compute_mode, inplace, resource),
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

        if (GROUP != "All") && inplace
            continue
        end

        @testset "$(typeof(adb).name.name)" for adb in adb_list
            @testset "Loss" begin
                @testset "ps" begin
                    @test !isnothing(AbstractDifferentiation.gradient(adb, diff_loss, ps))
                end
                @testset "x" begin
                    @test !isnothing(AbstractDifferentiation.gradient(adb, diff2_loss, r)) broken =
                        (GROUP != "All") &&
                        adb isa AbstractDifferentiation.ReverseDiffBackend &&
                        compute_mode <: ContinuousNormalizingFlows.MatrixMode &&
                        VERSION >= v"1.10"
                end
            end
        end
        @testset "$(typeof(adtype).name.name)" for adtype in adtypes
            @testset "Loss" begin
                @testset "ps" begin
                    @test !isnothing(
                        DifferentiationInterface.gradient(diff_loss, adtype, ps),
                    )
                end
                @testset "x" begin
                    @test !isnothing(
                        DifferentiationInterface.gradient(diff2_loss, adtype, r),
                    ) broken =
                        (GROUP != "All") &&
                        adtype isa ADTypes.AutoReverseDiff &&
                        compute_mode <: ContinuousNormalizingFlows.MatrixMode &&
                        VERSION >= v"1.10"
                end
            end
        end
    end
end

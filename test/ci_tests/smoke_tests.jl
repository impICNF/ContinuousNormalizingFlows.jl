Test.@testset verbose = true showtiming = true failfast = false "Smoke Tests" begin
    mts = Type{<:ContinuousNormalizingFlows.AbstractICNF}[ContinuousNormalizingFlows.ICNF]
    omodes = ContinuousNormalizingFlows.Mode[
        ContinuousNormalizingFlows.TrainMode{true}(),
        ContinuousNormalizingFlows.TestMode{true}(),
    ]
    conds, inplaces = if GROUP == "SmokeXOut"
        Bool[false], Bool[false]
    elseif GROUP == "SmokeXIn"
        Bool[false], Bool[true]
    elseif GROUP == "SmokeXYOut"
        Bool[true], Bool[false]
    elseif GROUP == "SmokeXYIn"
        Bool[true], Bool[true]
    else
        Bool[false, true], Bool[false, true]
    end
    planars = Bool[false, true]
    nvars_ = Int[2]
    ndata_ = Int[4]
    data_types = Type{<:AbstractFloat}[Float32]
    devices = MLDataDevices.AbstractDevice[MLDataDevices.cpu_device()]
    adtypes = ADTypes.AbstractADType[ADTypes.AutoZygote(),
    # ADTypes.AutoForwardDiff(),
    # ADTypes.AutoEnzyme(;
    #     mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    #     function_annotation = Enzyme.Const,
    # ),
    # ADTypes.AutoEnzyme(;
    #     mode = Enzyme.set_runtime_activity(Enzyme.Forward),
    #     function_annotation = Enzyme.Const,
    # ),
    ]
    compute_modes = ContinuousNormalizingFlows.ComputeMode[
        ContinuousNormalizingFlows.LuxVecJacMatrixMode(ADTypes.AutoZygote()),
        ContinuousNormalizingFlows.DIVecJacMatrixMode(ADTypes.AutoZygote()),
        ContinuousNormalizingFlows.DIVecJacVectorMode(ADTypes.AutoZygote()),
        ContinuousNormalizingFlows.LuxJacVecMatrixMode(ADTypes.AutoForwardDiff()),
        ContinuousNormalizingFlows.DIJacVecMatrixMode(ADTypes.AutoForwardDiff()),
        ContinuousNormalizingFlows.DIJacVecVectorMode(ADTypes.AutoForwardDiff()),
        ContinuousNormalizingFlows.LuxVecJacMatrixMode(
            ADTypes.AutoEnzyme(;
                mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                function_annotation = Enzyme.Const,
            ),
        ),
        ContinuousNormalizingFlows.DIVecJacMatrixMode(
            ADTypes.AutoEnzyme(;
                mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                function_annotation = Enzyme.Const,
            ),
        ),
        ContinuousNormalizingFlows.DIVecJacVectorMode(
            ADTypes.AutoEnzyme(;
                mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                function_annotation = Enzyme.Const,
            ),
        ),
        ContinuousNormalizingFlows.LuxJacVecMatrixMode(
            ADTypes.AutoEnzyme(;
                mode = Enzyme.set_runtime_activity(Enzyme.Forward),
                function_annotation = Enzyme.Const,
            ),
        ),
        ContinuousNormalizingFlows.DIJacVecMatrixMode(
            ADTypes.AutoEnzyme(;
                mode = Enzyme.set_runtime_activity(Enzyme.Forward),
                function_annotation = Enzyme.Const,
            ),
        ),
        ContinuousNormalizingFlows.DIJacVecVectorMode(
            ADTypes.AutoEnzyme(;
                mode = Enzyme.set_runtime_activity(Enzyme.Forward),
                function_annotation = Enzyme.Const,
            ),
        ),
    ]

    Test.@testset verbose = true showtiming = true failfast = false "$device | $data_type | $compute_mode | ndata = $ndata | nvars = $nvars | inplace = $inplace | cond = $cond | planar = $planar | $omode | $mt" for device in
                                                                                                                                                                                                                       devices,
        data_type in data_types,
        compute_mode in compute_modes,
        ndata in ndata_,
        nvars in nvars_,
        inplace in inplaces,
        cond in conds,
        planar in planars,
        omode in omodes,
        mt in mts

        data_dist =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
        data_dist2 =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (4, 2))...)
        if compute_mode isa ContinuousNormalizingFlows.VectorMode
            r = convert.(data_type, rand(data_dist, nvars))
            r2 = convert.(data_type, rand(data_dist2, nvars))
        elseif compute_mode isa ContinuousNormalizingFlows.MatrixMode
            r = convert.(data_type, rand(data_dist, nvars, ndata))
            r2 = convert.(data_type, rand(data_dist2, nvars, ndata))
        end
        df = DataFrames.DataFrame(transpose(r), :auto)
        df2 = DataFrames.DataFrame(transpose(r2), :auto)

        nn = ifelse(
            cond,
            ifelse(
                planar,
                Lux.Chain(
                    ContinuousNormalizingFlows.PlanarLayer(nvars * 2, tanh; n_cond = nvars),
                ),
                Lux.Chain(Lux.Dense(nvars * 3 => nvars * 2, tanh)),
            ),
            ifelse(
                planar,
                Lux.Chain(ContinuousNormalizingFlows.PlanarLayer(nvars * 2, tanh)),
                Lux.Chain(Lux.Dense(nvars * 2 => nvars * 2, tanh)),
            ),
        )
        icnf = ContinuousNormalizingFlows.construct(
            mt,
            nn,
            nvars,
            nvars;
            data_type,
            compute_mode,
            inplace,
            cond,
            device,
            steer_rate = convert(data_type, 1.0e-1),
            λ₁ = convert(data_type, 1.0e-2),
            λ₂ = convert(data_type, 1.0e-2),
            λ₃ = convert(data_type, 1.0e-2),
            sol_kwargs = (;
                save_everystep = false,
                alg = OrdinaryDiffEqDefault.DefaultODEAlgorithm(),
                sensealg = SciMLSensitivity.GaussAdjoint(),
            ),
        )
        ps, st = LuxCore.setup(icnf.rng, icnf)
        ps = ComponentArrays.ComponentArray(ps)
        r = icnf.device(r)
        r2 = icnf.device(r2)
        ps = icnf.device(ps)
        st = icnf.device(st)

        if cond
            Test.@test !isnothing(
                ContinuousNormalizingFlows.inference(icnf, omode, r, r2, ps, st),
            )
            if compute_mode isa ContinuousNormalizingFlows.MatrixMode
                Test.@test !isnothing(
                    ContinuousNormalizingFlows.generate(icnf, omode, r2, ps, st, ndata),
                )
            else
                Test.@test !isnothing(
                    ContinuousNormalizingFlows.generate(icnf, omode, r2, ps, st),
                )
            end

            Test.@test !isnothing(
                ContinuousNormalizingFlows.loss(icnf, omode, r, r2, ps, st),
            )
            Test.@test !isnothing(icnf((r, r2), ps, st))

            diff_loss = function (x::Any)
                return ContinuousNormalizingFlows.loss(icnf, omode, r, r2, x, st)
            end
            diff2_loss = function (x::Any)
                return ContinuousNormalizingFlows.loss(icnf, omode, x, r2, ps, st)
            end
        else
            Test.@test !isnothing(
                ContinuousNormalizingFlows.inference(icnf, omode, r, ps, st),
            )
            if compute_mode isa ContinuousNormalizingFlows.MatrixMode
                Test.@test !isnothing(
                    ContinuousNormalizingFlows.generate(icnf, omode, ps, st, ndata),
                )
            else
                Test.@test !isnothing(
                    ContinuousNormalizingFlows.generate(icnf, omode, ps, st),
                )
            end

            Test.@test !isnothing(ContinuousNormalizingFlows.loss(icnf, omode, r, ps, st))
            Test.@test !isnothing(icnf(r, ps, st))

            diff_loss = function (x::Any)
                return ContinuousNormalizingFlows.loss(icnf, omode, r, x, st)
            end
            diff2_loss = function (x::Any)
                return ContinuousNormalizingFlows.loss(icnf, omode, x, ps, st)
            end
        end

        if cond
            d = ContinuousNormalizingFlows.CondICNFDist(icnf, omode, r2, ps, st)
        else
            d = ContinuousNormalizingFlows.ICNFDist(icnf, omode, ps, st)
        end

        Test.@test !isnothing(Distributions.logpdf(d, r))
        Test.@test !isnothing(Distributions.pdf(d, r))
        Test.@test !isnothing(rand(d))
        Test.@test !isnothing(rand(d, ndata))

        Test.@testset verbose = true showtiming = true failfast = false "$adtype on loss" for adtype in
                                                                                              adtypes

            Test.@test !isnothing(DifferentiationInterface.gradient(diff_loss, adtype, ps)) broken =
                compute_mode.adback isa ADTypes.AutoEnzyme{<:Enzyme.ForwardMode} && (
                    omode isa ContinuousNormalizingFlows.TrainMode || (
                        omode isa ContinuousNormalizingFlows.TestMode &&
                        compute_mode isa ContinuousNormalizingFlows.VectorMode
                    )
                )
            Test.@test !isnothing(DifferentiationInterface.gradient(diff2_loss, adtype, r)) broken =
                compute_mode.adback isa ADTypes.AutoEnzyme{<:Enzyme.ForwardMode} && (
                    omode isa ContinuousNormalizingFlows.TrainMode || (
                        omode isa ContinuousNormalizingFlows.TestMode &&
                        compute_mode isa ContinuousNormalizingFlows.VectorMode
                    )
                )

            if cond
                model = ContinuousNormalizingFlows.CondICNFModel(
                    icnf;
                    adtype,
                    batchsize = 0,
                    sol_kwargs = (; epochs = 2),
                )
                mach = MLJBase.machine(model, (df, df2))

                Test.@test !isnothing(MLJBase.fit!(mach)) broken =
                    compute_mode.adback isa ADTypes.AutoEnzyme{<:Enzyme.ForwardMode}
                Test.@test !isnothing(MLJBase.transform(mach, (df, df2))) broken =
                    compute_mode.adback isa ADTypes.AutoEnzyme{<:Enzyme.ForwardMode}
                Test.@test !isnothing(MLJBase.fitted_params(mach)) broken =
                    compute_mode.adback isa ADTypes.AutoEnzyme{<:Enzyme.ForwardMode}
                Test.@test !isnothing(MLJBase.serializable(mach)) broken =
                    compute_mode.adback isa ADTypes.AutoEnzyme{<:Enzyme.ForwardMode}

                Test.@test !isnothing(
                    ContinuousNormalizingFlows.CondICNFDist(mach, omode, r2),
                ) broken = compute_mode.adback isa ADTypes.AutoEnzyme{<:Enzyme.ForwardMode}
            else
                model = ContinuousNormalizingFlows.ICNFModel(
                    icnf;
                    adtype,
                    batchsize = 0,
                    sol_kwargs = (; epochs = 2),
                )
                mach = MLJBase.machine(model, df)

                Test.@test !isnothing(MLJBase.fit!(mach)) broken =
                    compute_mode.adback isa ADTypes.AutoEnzyme{<:Enzyme.ForwardMode}
                Test.@test !isnothing(MLJBase.transform(mach, df)) broken =
                    compute_mode.adback isa ADTypes.AutoEnzyme{<:Enzyme.ForwardMode}
                Test.@test !isnothing(MLJBase.fitted_params(mach)) broken =
                    compute_mode.adback isa ADTypes.AutoEnzyme{<:Enzyme.ForwardMode}
                Test.@test !isnothing(MLJBase.serializable(mach)) broken =
                    compute_mode.adback isa ADTypes.AutoEnzyme{<:Enzyme.ForwardMode}

                Test.@test !isnothing(ContinuousNormalizingFlows.ICNFDist(mach, omode)) broken =
                    compute_mode.adback isa ADTypes.AutoEnzyme{<:Enzyme.ForwardMode}
            end
        end
    end
end

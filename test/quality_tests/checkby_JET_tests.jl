Test.@testset verbose = true showtiming = true failfast = false "CheckByJET" begin
    JET.test_package(
        ContinuousNormalizingFlows;
        target_modules = (ContinuousNormalizingFlows,),
    )

    omodes = ContinuousNormalizingFlows.Mode[
        ContinuousNormalizingFlows.TrainMode{true}(),
        ContinuousNormalizingFlows.TestMode(),
    ]
    conditioneds = Bool[false, true]
    inplaces = Bool[false, true]
    planars = Bool[false, true]
    devices = MLDataDevices.AbstractDevice[MLDataDevices.cpu_device()]
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

    Test.@testset verbose = true showtiming = true failfast = false "$device | $compute_mode | $omode | inplace = $inplace | conditioned = $conditioned | planar = $planar" for device in
                                                                                                                                                                                devices,
        compute_mode in compute_modes,
        omode in omodes,
        inplace in inplaces,
        conditioned in conditioneds,
        planar in planars

        ndata = 4
        ndimensions = 2
        data_dist = Distributions.Beta{Float32}(2.0f0, 4.0f0)
        data_dist2 = Distributions.Beta{Float32}(2.0f0, 4.0f0)
        if compute_mode isa ContinuousNormalizingFlows.VectorMode
            r = convert.(Float32, rand(data_dist, ndimensions))
            r2 = convert.(Float32, rand(data_dist2, ndimensions))
        elseif compute_mode isa ContinuousNormalizingFlows.MatrixMode
            r = convert.(Float32, rand(data_dist, ndimensions, ndata))
            r2 = convert.(Float32, rand(data_dist2, ndimensions, ndata))
        end
        nvariables = size(r, 1)

        icnf = ifelse(
            planar,
            ContinuousNormalizingFlows.ICNF(;
                nn = ifelse(
                    conditioned,
                    Lux.Chain(
                        ContinuousNormalizingFlows.PlanarLayer(
                            nvariables * 3 + 1 => nvariables * 2,
                            tanh,
                        ),
                    ),
                    Lux.Chain(
                        ContinuousNormalizingFlows.PlanarLayer(
                            nvariables * 2 + 1 => nvariables * 2,
                            tanh,
                        ),
                    ),
                ),
                nvariables,
                nconditions = ifelse(conditioned, nvariables, 0),
                inplace,
                compute_mode,
                device,
            ),
            ContinuousNormalizingFlows.ICNF(;
                nvariables,
                nconditions = ifelse(conditioned, nvariables, 0),
                inplace,
                compute_mode,
                device,
            ),
        )
        ps, st = LuxCore.setup(icnf.rng, icnf)
        ps = ComponentArrays.ComponentArray(ps)
        r = icnf.device(r)
        r2 = icnf.device(r2)
        ps = icnf.device(ps)
        st = icnf.device(st)

        if conditioned
            ContinuousNormalizingFlows.loss(icnf, omode, r, r2, ps, st)
            JET.@test_call target_modules = (ContinuousNormalizingFlows,) ContinuousNormalizingFlows.loss(
                icnf,
                omode,
                r,
                r2,
                ps,
                st,
            )
            JET.@test_opt target_modules = (ContinuousNormalizingFlows,) ContinuousNormalizingFlows.loss(
                icnf,
                omode,
                r,
                r2,
                ps,
                st,
            )
        else
            ContinuousNormalizingFlows.loss(icnf, omode, r, ps, st)
            JET.@test_call target_modules = (ContinuousNormalizingFlows,) ContinuousNormalizingFlows.loss(
                icnf,
                omode,
                r,
                ps,
                st,
            )
            JET.@test_opt target_modules = (ContinuousNormalizingFlows,) ContinuousNormalizingFlows.loss(
                icnf,
                omode,
                r,
                ps,
                st,
            )
        end
    end
end

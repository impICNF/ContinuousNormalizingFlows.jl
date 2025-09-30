Test.@testset "CheckByJET" begin
    JET.test_package(
        ContinuousNormalizingFlows;
        target_modules = [ContinuousNormalizingFlows],
        mode = :typo,
    )

    mts = Type{<:ContinuousNormalizingFlows.AbstractICNF}[ContinuousNormalizingFlows.ICNF]
    omodes = ContinuousNormalizingFlows.Mode[
        ContinuousNormalizingFlows.TrainMode{true}(),
        ContinuousNormalizingFlows.TestMode{true}(),
    ]
    conds = Bool[false, true]
    inplaces = Bool[false, true]
    planars = Bool[false, true]
    nvars_ = Int[2]
    ndata_ = Int[4]
    data_types = Type{<:AbstractFloat}[Float32]
    devices = MLDataDevices.AbstractDevice[MLDataDevices.cpu_device()]
    compute_modes = ContinuousNormalizingFlows.ComputeMode[
        ContinuousNormalizingFlows.LuxVecJacMatrixMode(ADTypes.AutoZygote()),
        ContinuousNormalizingFlows.DIVecJacVectorMode(ADTypes.AutoZygote()),
        ContinuousNormalizingFlows.DIVecJacMatrixMode(ADTypes.AutoZygote()),
        ContinuousNormalizingFlows.LuxJacVecMatrixMode(ADTypes.AutoForwardDiff()),
        ContinuousNormalizingFlows.DIJacVecVectorMode(ADTypes.AutoForwardDiff()),
        ContinuousNormalizingFlows.DIJacVecMatrixMode(ADTypes.AutoForwardDiff()),
        ContinuousNormalizingFlows.DIVecJacVectorMode(
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
        ContinuousNormalizingFlows.DIJacVecVectorMode(
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
    ]

    Test.@testset "$device | $data_type | $compute_mode | ndata = $ndata | nvars = $nvars | inplace = $inplace | cond = $cond | planar = $planar | $omode | $mt" for device in
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
                sensealg = SciMLSensitivity.InterpolatingAdjoint(),
            ),
        )
        ps, st = LuxCore.setup(icnf.rng, icnf)
        ps = ComponentArrays.ComponentArray(ps)
        r = icnf.device(r)
        r2 = icnf.device(r2)
        ps = icnf.device(ps)
        st = icnf.device(st)

        if cond
            ContinuousNormalizingFlows.loss(icnf, omode, r, r2, ps, st)
            JET.test_call(
                ContinuousNormalizingFlows.loss,
                Base.typesof(icnf, omode, r, r2, ps, st);
                target_modules = [ContinuousNormalizingFlows],
                mode = :typo,
            )
            JET.test_opt(
                ContinuousNormalizingFlows.loss,
                Base.typesof(icnf, omode, r, r2, ps, st);
                target_modules = [ContinuousNormalizingFlows],
            )
        else
            ContinuousNormalizingFlows.loss(icnf, omode, r, ps, st)
            JET.test_call(
                ContinuousNormalizingFlows.loss,
                Base.typesof(icnf, omode, r, ps, st);
                target_modules = [ContinuousNormalizingFlows],
                mode = :typo,
            )
            JET.test_opt(
                ContinuousNormalizingFlows.loss,
                Base.typesof(icnf, omode, r, ps, st);
                target_modules = [ContinuousNormalizingFlows],
            )
        end
    end
end

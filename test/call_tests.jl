Test.@testset "Call Tests" begin
    mts = if GROUP == "RNODE"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[ContinuousNormalizingFlows.RNODE]
    elseif GROUP == "FFJORD"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[ContinuousNormalizingFlows.FFJORD]
    elseif GROUP == "Planar"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[ContinuousNormalizingFlows.Planar]
    elseif GROUP == "CondRNODE"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[ContinuousNormalizingFlows.CondRNODE]
    elseif GROUP == "CondFFJORD"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[ContinuousNormalizingFlows.CondFFJORD]
    elseif GROUP == "CondPlanar"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[ContinuousNormalizingFlows.CondPlanar]
    else
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[
            ContinuousNormalizingFlows.RNODE,
            ContinuousNormalizingFlows.FFJORD,
            ContinuousNormalizingFlows.Planar,
            ContinuousNormalizingFlows.CondRNODE,
            ContinuousNormalizingFlows.CondFFJORD,
            ContinuousNormalizingFlows.CondPlanar,
        ]
    end
    omodes = ContinuousNormalizingFlows.Mode[
        ContinuousNormalizingFlows.TrainMode(),
        ContinuousNormalizingFlows.TestMode(),
    ]
    ndata_ = Int[4]
    nvars_ = Int[2]
    aug_steers = Bool[false, true]
    inplaces = Bool[false, true]
    adtypes = ADTypes.AbstractADType[
        ADTypes.AutoZygote(),
        ADTypes.AutoForwardDiff(),
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
        ContinuousNormalizingFlows.LuxJacVecMatrixMode(ADTypes.AutoForwardDiff()),
        ContinuousNormalizingFlows.DIVecJacVectorMode(ADTypes.AutoZygote()),
        ContinuousNormalizingFlows.DIJacVecVectorMode(ADTypes.AutoForwardDiff()),
        ContinuousNormalizingFlows.DIVecJacMatrixMode(ADTypes.AutoZygote()),
        ContinuousNormalizingFlows.DIJacVecMatrixMode(ADTypes.AutoForwardDiff()),
        # ContinuousNormalizingFlows.DIVecJacVectorMode(
        #     ADTypes.AutoEnzyme(;
        #         mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
        #         function_annotation = Enzyme.Const,
        #     ),
        # ),
        # ContinuousNormalizingFlows.DIJacVecVectorMode(
        #     ADTypes.AutoEnzyme(;
        #         mode = Enzyme.set_runtime_activity(Enzyme.Forward),
        #         function_annotation = Enzyme.Const,
        #     ),
        # ),
        # ContinuousNormalizingFlows.DIVecJacMatrixMode(
        #     ADTypes.AutoEnzyme(;
        #         mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
        #         function_annotation = Enzyme.Const,
        #     ),
        # ),
        # ContinuousNormalizingFlows.DIJacVecMatrixMode(
        #     ADTypes.AutoEnzyme(;
        #         mode = Enzyme.set_runtime_activity(Enzyme.Forward),
        #         function_annotation = Enzyme.Const,
        #     ),
        # ),
    ]
    data_types = Type{<:AbstractFloat}[Float32]
    devices = MLDataDevices.AbstractDevice[MLDataDevices.cpu_device()]

    Test.@testset "$device | $data_type | $compute_mode | inplace = $inplace | aug & steer = $aug_steer | nvars = $nvars | ndata = $ndata | $omode | $mt" for device in
                                                                                                                                                              devices,
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
        if compute_mode isa ContinuousNormalizingFlows.VectorMode
            r = convert.(data_type, rand(data_dist, nvars))
            r2 = convert.(data_type, rand(data_dist2, nvars))
        elseif compute_mode isa ContinuousNormalizingFlows.MatrixMode
            r = convert.(data_type, rand(data_dist, nvars, ndata))
            r2 = convert.(data_type, rand(data_dist2, nvars, ndata))
        end

        nn = ifelse(
            mt <: Union{
                ContinuousNormalizingFlows.CondRNODE,
                ContinuousNormalizingFlows.CondFFJORD,
                ContinuousNormalizingFlows.CondPlanar,
            },
            ifelse(
                mt <: ContinuousNormalizingFlows.CondPlanar,
                ifelse(
                    aug_steer,
                    Lux.Chain(
                        ContinuousNormalizingFlows.PlanarLayer(
                            nvars * 2,
                            tanh;
                            n_cond = nvars,
                        ),
                    ),
                    Lux.Chain(
                        ContinuousNormalizingFlows.PlanarLayer(nvars, tanh; n_cond = nvars),
                    ),
                ),
                ifelse(
                    aug_steer,
                    Lux.Chain(Lux.Dense(nvars * 3 => nvars * 2, tanh)),
                    Lux.Chain(Lux.Dense(nvars * 2 => nvars, tanh)),
                ),
            ),
            ifelse(
                mt <: ContinuousNormalizingFlows.Planar,
                ifelse(
                    aug_steer,
                    Lux.Chain(ContinuousNormalizingFlows.PlanarLayer(nvars * 2, tanh)),
                    Lux.Chain(ContinuousNormalizingFlows.PlanarLayer(nvars, tanh)),
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
            ContinuousNormalizingFlows.construct(
                mt,
                nn,
                nvars,
                nvars;
                data_type,
                compute_mode,
                inplace,
                device,
                steer_rate = convert(data_type, 1.0e-1),
                λ₃ = convert(data_type, 1.0e-2),
            ),
            ContinuousNormalizingFlows.construct(
                mt,
                nn,
                nvars;
                data_type,
                compute_mode,
                inplace,
                device,
            ),
        )
        ps, st = Lux.setup(icnf.rng, icnf)
        ps = ComponentArrays.ComponentArray(ps)
        r = device(r)
        r2 = device(r2)
        ps = device(ps)
        st = device(st)
        if mt <: Union{
            ContinuousNormalizingFlows.CondRNODE,
            ContinuousNormalizingFlows.CondFFJORD,
            ContinuousNormalizingFlows.CondPlanar,
        }
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

            diff_loss = function (x)
                return ContinuousNormalizingFlows.loss(icnf, omode, r, r2, x, st)
            end
            diff2_loss = function (x)
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

            diff_loss = function (x)
                return ContinuousNormalizingFlows.loss(icnf, omode, r, x, st)
            end
            diff2_loss = function (x)
                return ContinuousNormalizingFlows.loss(icnf, omode, x, ps, st)
            end
        end

        if mt <: Union{
            ContinuousNormalizingFlows.CondRNODE,
            ContinuousNormalizingFlows.CondFFJORD,
            ContinuousNormalizingFlows.CondPlanar,
        }
            d = ContinuousNormalizingFlows.CondICNFDist(icnf, omode, r2, ps, st)
        else
            d = ContinuousNormalizingFlows.ICNFDist(icnf, omode, ps, st)
        end

        Test.@test !isnothing(Distributions.logpdf(d, r))
        Test.@test !isnothing(Distributions.pdf(d, r))
        Test.@test !isnothing(rand(d))
        Test.@test !isnothing(rand(d, ndata))

        Test.@testset "$adtype on loss" for adtype in adtypes
            Test.@test !isnothing(DifferentiationInterface.gradient(diff_loss, adtype, ps))
            Test.@test !isnothing(DifferentiationInterface.gradient(diff2_loss, adtype, r))
        end
    end
end

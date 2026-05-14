Test.@testset verbose = true showtiming = true failfast = false "CheckByJET" begin
    JET.test_package(
        ContinuousNormalizingFlows;
        target_modules = (ContinuousNormalizingFlows,),
    )
    Test.@testset verbose = true showtiming = true failfast = false "$device | $compute_mode | $omode | inplace = $inplace | conditioned = $conditioned | planar = $planar" for device in
                                                                                                                                                                                devices,
        compute_mode in compute_modes,
        omode in omodes,
        inplace in inplaces,
        conditioned in conditioneds,
        planar in planars

        ndata = 4
        ndimensions = 2
        data_dist = Distributions.Beta(2.0, 4.0)
        data_dist2 = Distributions.Beta(2.0, 4.0)
        if compute_mode isa ContinuousNormalizingFlows.VectorMode
            r = rand(data_dist, ndimensions)
            r2 = rand(data_dist2, ndimensions)
        elseif compute_mode isa ContinuousNormalizingFlows.MatrixMode
            r = rand(data_dist, ndimensions, ndata)
            r2 = rand(data_dist2, ndimensions, ndata)
        end
        nvariables = size(r, 1)

        icnf = ifelse(
            planar,
            ContinuousNormalizingFlows.ICNF(;
                nn = ifelse(
                    conditioned,
                    Lux.Chain(
                        ContinuousNormalizingFlows.PlanarLayer(
                            nvariables * 3 + 2 => nvariables * 2 + 1,
                            tanh,
                        ),
                    ),
                    Lux.Chain(
                        ContinuousNormalizingFlows.PlanarLayer(
                            nvariables * 2 + 2 => nvariables * 2 + 1,
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

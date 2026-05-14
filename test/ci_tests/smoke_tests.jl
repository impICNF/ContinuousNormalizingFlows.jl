Test.@testset verbose = true showtiming = true failfast = false "Smoke Tests" begin
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
        df = DataFrames.DataFrame(permutedims(r), :auto)
        df2 = DataFrames.DataFrame(permutedims(r2), :auto)
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

        if conditioned
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

            Test.@test !isnothing(DifferentiationInterface.gradient(diff_loss, adtype, ps))
            Test.@test !isnothing(DifferentiationInterface.gradient(diff2_loss, adtype, r))

            if conditioned
                model = ContinuousNormalizingFlows.CondICNFModel(; icnf, adtype)
                mach = MLJBase.machine(model, (df, df2))

                Test.@test !isnothing(MLJBase.fit!(mach))
                Test.@test !isnothing(MLJBase.transform(mach, (df, df2)))
                Test.@test !isnothing(MLJBase.fitted_params(mach))
                Test.@test !isnothing(MLJBase.serializable(mach))

                Test.@test !isnothing(
                    ContinuousNormalizingFlows.CondICNFDist(mach, omode, r2),
                )
            else
                model = ContinuousNormalizingFlows.ICNFModel(; icnf, adtype)
                mach = MLJBase.machine(model, df)

                Test.@test !isnothing(MLJBase.fit!(mach))
                Test.@test !isnothing(MLJBase.transform(mach, df))
                Test.@test !isnothing(MLJBase.fitted_params(mach))
                Test.@test !isnothing(MLJBase.serializable(mach))

                Test.@test !isnothing(ContinuousNormalizingFlows.ICNFDist(mach, omode))
            end
        end
    end
end

Test.@testset verbose = true showtiming = true failfast = false "Speed Tests" begin
    compute_modes = ContinuousNormalizingFlows.ComputeMode[
        ContinuousNormalizingFlows.LuxVecJacMatrixMode(ADTypes.AutoZygote()),
        ContinuousNormalizingFlows.DIVecJacMatrixMode(ADTypes.AutoZygote()),
        ContinuousNormalizingFlows.LuxJacVecMatrixMode(ADTypes.AutoForwardDiff()),
        ContinuousNormalizingFlows.DIJacVecMatrixMode(ADTypes.AutoForwardDiff()),
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
    ]

    Test.@testset verbose = true showtiming = true failfast = false "$compute_mode" for compute_mode in
                                                                                        compute_modes

        @show compute_mode

        rng = StableRNGs.StableRNG(1)
        ndata = 2^10
        ndimension = 1
        data_dist = Distributions.Beta{Float32}(2.0f0, 4.0f0)
        r = rand(rng, data_dist, ndimension, ndata)
        r = convert.(Float32, r)

        nvars = size(r, 1)
        naugs = nvars + 1
        n_in = nvars + naugs

        nn = Lux.Chain(
            Lux.Dense(n_in => (2 * n_in + 1), tanh),
            Lux.Dense((2 * n_in + 1) => n_in, tanh),
        )

        icnf = ContinuousNormalizingFlows.ICNF(;
            nn,
            nvars,
            naugmented = naugs,
            rng,
            compute_mode,
        )

        df = DataFrames.DataFrame(transpose(r), :auto)
        model = ContinuousNormalizingFlows.ICNFModel(;
            icnf,
            batchsize = 0,
            sol_kwargs = (; epochs = 5),
        )

        mach = MLJBase.machine(model, df)
        MLJBase.fit!(mach)

        @show only(MLJBase.report(mach).stats).time

        d = ContinuousNormalizingFlows.ICNFDist(
            mach,
            ContinuousNormalizingFlows.TestMode{true}(),
        )
        actual_pdf = Distributions.pdf.(data_dist, r)
        estimated_pdf = Distributions.pdf(d, r)

        mad_ = Distances.meanad(estimated_pdf, actual_pdf)
        msd_ = Distances.msd(estimated_pdf, actual_pdf)
        tv_dis = Distances.totalvariation(estimated_pdf, actual_pdf) / ndata

        @show mad_
        @show msd_
        @show tv_dis
        Test.@test true
    end
end

Test.@testset "Speed Tests" begin
    compute_modes = ContinuousNormalizingFlows.ComputeMode[
        ContinuousNormalizingFlows.LuxVecJacMatrixMode(ADTypes.AutoZygote()),
        ContinuousNormalizingFlows.DIVecJacMatrixMode(ADTypes.AutoZygote()),
        ContinuousNormalizingFlows.DIVecJacMatrixMode(
            ADTypes.AutoEnzyme(;
                mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                function_annotation = Enzyme.Const,
            ),
        ),
        ContinuousNormalizingFlows.DIJacVecMatrixMode(
            ADTypes.AutoEnzyme(;
                mode = Enzyme.set_runtime_activity(Enzyme.Forward),
                function_annotation = Enzyme.Const,
            ),
        ),
        ContinuousNormalizingFlows.DIJacVecMatrixMode(ADTypes.AutoForwardDiff()),
        ContinuousNormalizingFlows.LuxJacVecMatrixMode(ADTypes.AutoForwardDiff()),
    ]

    Test.@testset "$compute_mode" for compute_mode in compute_modes
        @show compute_mode

        rng = StableRNGs.StableRNG(1)
        nvars = 1
        naugs = nvars
        n_in = nvars + naugs
        n = 2^10
        nn = Lux.Chain(Lux.Dense(n_in => 3 * n_in, tanh), Lux.Dense(3 * n_in => n_in, tanh))

        icnf = ContinuousNormalizingFlows.construct(
            ContinuousNormalizingFlows.ICNF,
            nn,
            nvars,
            naugs;
            compute_mode,
            tspan = (0.0f0, 1.0f0),
            steer_rate = 1.0f-1,
            λ₁ = 1.0f-2,
            λ₂ = 1.0f-2,
            λ₃ = 1.0f-2,
            rng,
            sol_kwargs = (;
                save_everystep = false,
                alg = OrdinaryDiffEqDefault.DefaultODEAlgorithm(),
                sensealg = SciMLSensitivity.InterpolatingAdjoint(),
            ),
        )

        data_dist = Distributions.Beta{Float32}(2.0f0, 4.0f0)
        r = rand(icnf.rng, data_dist, nvars, n)
        r = convert.(Float32, r)

        df = DataFrames.DataFrame(transpose(r), :auto)
        model = ContinuousNormalizingFlows.ICNFModel(icnf; batch_size = 0, n_epochs = 5)

        mach = MLJBase.machine(model, df)
        MLJBase.fit!(mach)

        @show only(MLJBase.report(mach).stats).time

        d = ContinuousNormalizingFlows.ICNFDist(mach, ContinuousNormalizingFlows.TestMode())
        actual_pdf = Distributions.pdf.(data_dist, r)
        estimated_pdf = Distributions.pdf(d, r)

        mad_ = Distances.meanad(estimated_pdf, actual_pdf)
        msd_ = Distances.msd(estimated_pdf, actual_pdf)
        tv_dis = Distances.totalvariation(estimated_pdf, actual_pdf) / n

        @show mad_
        @show msd_
        @show tv_dis
        Test.@test true
    end
end

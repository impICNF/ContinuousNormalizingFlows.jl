Test.@testset verbose = true showtiming = true failfast = false "Regression Tests" begin
    rng = StableRNGs.StableRNG(1)
    ndata = 2^10
    ndimension = 1
    data_dist = Distributions.Beta{Float32}(2.0f0, 4.0f0)
    r = rand(rng, data_dist, ndimension, ndata)
    r = convert.(Float32, r)

    nvars = size(r, 1)
    naugs = nvars
    n_in = nvars + naugs

    nn = Lux.Chain(Lux.Dense(n_in => 3 * n_in, tanh), Lux.Dense(3 * n_in => n_in, tanh))

    icnf = ContinuousNormalizingFlows.construct(
        ContinuousNormalizingFlows.ICNF,
        nn,
        nvars,
        naugs;
        compute_mode = ContinuousNormalizingFlows.LuxVecJacMatrixMode(ADTypes.AutoZygote()),
        tspan = (0.0f0, 1.0f0),
        steer_rate = 1.0f-1,
        λ₁ = 1.0f-2,
        λ₂ = 1.0f-2,
        λ₃ = 1.0f-2,
        rng,
        sol_kwargs = (;
            save_everystep = false,
            alg = OrdinaryDiffEqDefault.DefaultODEAlgorithm(),
            sensealg = SciMLSensitivity.GaussAdjoint(),
        ),
    )

    df = DataFrames.DataFrame(transpose(r), :auto)
    model = ContinuousNormalizingFlows.ICNFModel(
        icnf;
        batchsize = 0,
        sol_kwargs = (; epochs = 300),
    )

    mach = MLJBase.machine(model, df)
    MLJBase.fit!(mach)

    d = ContinuousNormalizingFlows.ICNFDist(
        mach,
        ContinuousNormalizingFlows.TestMode{true}(),
    )
    actual_pdf = Distributions.pdf.(data_dist, r)
    estimated_pdf = Distributions.pdf(d, r)

    mad_ = Distances.meanad(estimated_pdf, actual_pdf)
    msd_ = Distances.msd(estimated_pdf, actual_pdf)
    tv_dis = Distances.totalvariation(estimated_pdf, actual_pdf) / ndata

    Test.@test mad_ <= 1.0f2
    Test.@test msd_ <= 1.0f2
    Test.@test tv_dis <= 1.0f2
end

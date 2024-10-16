Test.@testset "Regression Tests" begin
    rng = StableRNGs.StableRNG(1)
    nvars = 2^3
    naugs = nvars
    n_in = nvars + naugs
    n = 2^10
    nn = Lux.Chain(Lux.Dense(n_in => 3 * n_in, tanh), Lux.Dense(3 * n_in => n_in, tanh))

    icnf = ContinuousNormalizingFlows.construct(
        ContinuousNormalizingFlows.RNODE,
        nn,
        nvars,
        naugs;
        compute_mode = ContinuousNormalizingFlows.DIJacVecMatrixMode(
            ADTypes.AutoEnzyme(;
                mode = Enzyme.set_runtime_activity(Enzyme.Forward),
                function_annotation = Enzyme.Const,
            ),
        ),
        tspan = (0.0f0, 13.0f0),
        steer_rate = 1.0f-1,
        λ₃ = 1.0f-2,
        rng,
    )
    ps, st = Lux.setup(icnf.rng, icnf)
    ps = ComponentArrays.ComponentArray(ps)

    data_dist = Distributions.Beta{Float32}(2.0f0, 4.0f0)
    r = rand(icnf.rng, data_dist, nvars, n)
    r = convert.(Float32, r)

    df = DataFrames.DataFrame(transpose(r), :auto)
    model = ContinuousNormalizingFlows.ICNFModel(icnf)

    mach = MLJBase.machine(model, df)
    MLJBase.fit!(mach)

    d = ContinuousNormalizingFlows.ICNFDist(mach, ContinuousNormalizingFlows.TestMode())
    actual_pdf = Distributions.pdf.(data_dist, r)
    estimated_pdf = Distributions.pdf(d, r)

    mad_ = Distances.meanad(estimated_pdf, actual_pdf)
    msd_ = Distances.msd(estimated_pdf, actual_pdf)
    tv_dis = Distances.totalvariation(estimated_pdf, actual_pdf) / n

    Test.@test mad_ <= 1.0f-1
    Test.@test msd_ <= 1.0f-1
    Test.@test tv_dis <= 1.0f-1
end

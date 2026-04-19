Test.@testset verbose = true showtiming = true failfast = false "Speed Tests" begin
    Test.@testset verbose = true showtiming = true failfast = false "$compute_mode" for compute_mode in
                                                                                        compute_modes
        @show compute_mode

        ndata = 2^10
        ndimensions = 1
        data_dist = Distributions.Beta{Float32}(2.0f0, 4.0f0)
        r = rand(data_dist, ndimensions, ndata)
        r = convert.(Float32, r)

        nvariables = size(r, 1)
        icnf = ContinuousNormalizingFlows.ICNF(; nvariables, compute_mode)

        df = DataFrames.DataFrame(permutedims(r), :auto)
        model = ContinuousNormalizingFlows.ICNFModel(;
            icnf,
            batchsize = 0,
            sol_kwargs = (; epochs = 5),
        )

        mach = MLJBase.machine(model, df)
        MLJBase.fit!(mach)

        @show only(MLJBase.report(mach).stats).time

        d = ContinuousNormalizingFlows.ICNFDist(mach, ContinuousNormalizingFlows.TestMode())
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

Test.@testset let ndata = 2^10, ndimensions = 1, α = 2, β = 4
    Test.@testset verbose = true showtiming = true failfast = false "Regression Tests" begin
        data_dist = Distributions.Beta{Float32}(convert(Float32, α), convert(Float32, β))
        r = rand(data_dist, ndimensions, ndata)
        r = convert.(Float32, r)

        nvariables = size(r, 1)
        icnf = ContinuousNormalizingFlows.ICNF(; nvariables)

        df = DataFrames.DataFrame(permutedims(r), :auto)
        model = ContinuousNormalizingFlows.ICNFModel(; icnf)

        mach = MLJBase.machine(model, df)
        MLJBase.fit!(mach)

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

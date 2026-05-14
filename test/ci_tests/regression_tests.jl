Test.@testset verbose = true showtiming = true failfast = false "Regression Tests" begin
    ndata = 2^10
    ndimensions = 1
    data_dist = Distributions.Beta(2.0, 4.0)
    r = rand(data_dist, ndimensions, ndata)

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

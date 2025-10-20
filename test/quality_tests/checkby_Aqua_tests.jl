Test.@testset verbose = true showtiming = true failfast = false "CheckByAqua" begin
    Aqua.test_all(ContinuousNormalizingFlows)
end

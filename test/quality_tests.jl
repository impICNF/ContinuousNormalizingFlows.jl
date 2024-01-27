@testset "Quality" begin
    @testset "Method ambiguity" begin
        test_ambiguities(ContinuousNormalizingFlows)
    end
    Aqua.test_all(ContinuousNormalizingFlows; ambiguities = false)
end

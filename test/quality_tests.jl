@testset "Quality" begin
    @testset "Method ambiguity" begin
        Aqua.test_ambiguities(ContinuousNormalizingFlows)
    end
    Aqua.test_all(ContinuousNormalizingFlows; ambiguities = false)
end

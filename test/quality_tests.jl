Test.@testset "Quality" begin
    Test.@testset "Method ambiguity" begin
        Aqua.test_ambiguities(ContinuousNormalizingFlows)
    end
    Aqua.test_all(ContinuousNormalizingFlows; ambiguities = (GROUP == "All"))
end

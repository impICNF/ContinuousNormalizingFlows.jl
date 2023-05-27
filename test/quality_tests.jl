@testset "Quality" begin
    Aqua.test_ambiguities(ContinuousNormalizingFlows)
    Aqua.test_unbound_args(ContinuousNormalizingFlows)
    Aqua.test_undefined_exports(ContinuousNormalizingFlows)
    Aqua.test_project_extras(ContinuousNormalizingFlows)
    Aqua.test_stale_deps(ContinuousNormalizingFlows)
    Aqua.test_deps_compat(ContinuousNormalizingFlows)
    Aqua.test_project_toml_formatting(ContinuousNormalizingFlows)

    @test isempty(Test.detect_ambiguities(ContinuousNormalizingFlows; recursive = true))
    @test isempty(Test.detect_unbound_args(ContinuousNormalizingFlows; recursive = true))
end

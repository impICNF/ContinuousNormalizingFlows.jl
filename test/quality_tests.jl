@testset "Quality" begin
    Aqua.test_ambiguities(ContinuousNF)
    Aqua.test_unbound_args(ContinuousNF)
    Aqua.test_undefined_exports(ContinuousNF)
    Aqua.test_project_extras(ContinuousNF)
    Aqua.test_stale_deps(ContinuousNF)
    Aqua.test_deps_compat(ContinuousNF)
    Aqua.test_project_toml_formatting(ContinuousNF)

    @test isempty(Test.detect_ambiguities(ContinuousNF; recursive = true))
    @test isempty(Test.detect_unbound_args(ContinuousNF; recursive = true))
end

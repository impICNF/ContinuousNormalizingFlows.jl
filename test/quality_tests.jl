@testset "Quality" begin
    Aqua.test_ambiguities(ICNF)
    Aqua.test_unbound_args(ICNF)
    Aqua.test_undefined_exports(ICNF)
    Aqua.test_project_extras(ICNF)
    Aqua.test_stale_deps(ICNF)
    Aqua.test_deps_compat(ICNF)
    Aqua.test_project_toml_formatting(ICNF)

    @test isempty(Test.detect_ambiguities(ICNF; recursive = true))
    @test isempty(Test.detect_unbound_args(ICNF; recursive = true))
end

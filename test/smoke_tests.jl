@testset "Smoke Tests" begin
    if GROUP == "All" || GROUP == "Call"
        include("call_tests.jl")
    end
    if GROUP == "All" || GROUP == "Fit"
        include("fit_tests.jl")
    end
end

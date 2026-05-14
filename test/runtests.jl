import ADTypes,
    Aqua,
    ComponentArrays,
    DataFrames,
    DifferentiationInterface,
    Distances,
    Distributions,
    Enzyme,
    ExplicitImports,
    ForwardDiff,
    JET,
    Lux,
    LuxCore,
    MLDataDevices,
    MLJBase,
    Test,
    Zygote,
    ContinuousNormalizingFlows

GROUP = get(ENV, "GROUP", "All")
VIA_ZYGOTE = parse(Bool, get(ENV, "VIA_ZYGOTE", "true"))
VIA_FORWARDDIFF = parse(Bool, get(ENV, "VIA_FORWARDDIFF", "true"))
VIA_ENZYME = parse(Bool, get(ENV, "VIA_ENZYME", "false"))

omodes = ContinuousNormalizingFlows.Mode[
    ContinuousNormalizingFlows.TrainMode{true}(),
    ContinuousNormalizingFlows.TestMode(),
]
conditioneds, inplaces = if GROUP == "SmokeXOut"
    Bool[false], Bool[false]
elseif GROUP == "SmokeXIn"
    Bool[false], Bool[true]
elseif GROUP == "SmokeXYOut"
    Bool[true], Bool[false]
elseif GROUP == "SmokeXYIn"
    Bool[true], Bool[true]
else
    Bool[false, true], Bool[false, true]
end
planars = Bool[false, true]
devices = MLDataDevices.AbstractDevice[MLDataDevices.cpu_device()]
adtypes = ADTypes.AbstractADType[]
compute_modes = ContinuousNormalizingFlows.ComputeMode[]
if VIA_ZYGOTE
    adtypes = append!(adtypes, ADTypes.AbstractADType[ADTypes.AutoZygote()])
    compute_modes = append!(
        compute_modes,
        ContinuousNormalizingFlows.ComputeMode[
            ContinuousNormalizingFlows.LuxVecJacMatrixMode(ADTypes.AutoZygote()),
            ContinuousNormalizingFlows.DIVecJacMatrixMode(ADTypes.AutoZygote()),
            ContinuousNormalizingFlows.DIVecJacVectorMode(ADTypes.AutoZygote()),
        ],
    )
end
if VIA_FORWARDDIFF
    adtypes = append!(adtypes, ADTypes.AbstractADType[ADTypes.AutoForwardDiff()])
    compute_modes = append!(
        compute_modes,
        ContinuousNormalizingFlows.ComputeMode[
            ContinuousNormalizingFlows.LuxJacVecMatrixMode(ADTypes.AutoForwardDiff()),
            ContinuousNormalizingFlows.DIJacVecMatrixMode(ADTypes.AutoForwardDiff()),
            ContinuousNormalizingFlows.DIJacVecVectorMode(ADTypes.AutoForwardDiff()),
        ],
    )
end
if VIA_ENZYME
    adtypes = append!(
        adtypes,
        ADTypes.AbstractADType[
            ADTypes.AutoEnzyme(;
                mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                function_annotation = Enzyme.Const,
            ),
            ADTypes.AutoEnzyme(;
                mode = Enzyme.set_runtime_activity(Enzyme.Forward),
                function_annotation = Enzyme.Const,
            ),
        ],
    )
    compute_modes = append!(
        compute_modes,
        ContinuousNormalizingFlows.ComputeMode[
            ContinuousNormalizingFlows.LuxVecJacMatrixMode(
                ADTypes.AutoEnzyme(;
                    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                    function_annotation = Enzyme.Const,
                ),
            ),
            ContinuousNormalizingFlows.DIVecJacMatrixMode(
                ADTypes.AutoEnzyme(;
                    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                    function_annotation = Enzyme.Const,
                ),
            ),
            ContinuousNormalizingFlows.DIVecJacVectorMode(
                ADTypes.AutoEnzyme(;
                    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                    function_annotation = Enzyme.Const,
                ),
            ),
            ContinuousNormalizingFlows.LuxJacVecMatrixMode(
                ADTypes.AutoEnzyme(;
                    mode = Enzyme.set_runtime_activity(Enzyme.Forward),
                    function_annotation = Enzyme.Const,
                ),
            ),
            ContinuousNormalizingFlows.DIJacVecMatrixMode(
                ADTypes.AutoEnzyme(;
                    mode = Enzyme.set_runtime_activity(Enzyme.Forward),
                    function_annotation = Enzyme.Const,
                ),
            ),
            ContinuousNormalizingFlows.DIJacVecVectorMode(
                ADTypes.AutoEnzyme(;
                    mode = Enzyme.set_runtime_activity(Enzyme.Forward),
                    function_annotation = Enzyme.Const,
                ),
            ),
        ],
    )
end

Test.@testset verbose = true showtiming = true failfast = false "Overall" begin
    if GROUP == "All" || GROUP in ["SmokeXOut", "SmokeXIn", "SmokeXYOut", "SmokeXYIn"]
        include("ci_tests/smoke_tests.jl")
    end

    if GROUP == "All" || GROUP == "Regression"
        include("ci_tests/regression_tests.jl")
    end

    if GROUP == "All" || GROUP == "CheckByAqua"
        include("quality_tests/checkby_Aqua_tests.jl")
    end

    if GROUP == "All" || GROUP == "CheckByExplicitImports"
        include("quality_tests/checkby_ExplicitImports_tests.jl")
    end

    if GROUP == "All" || GROUP == "CheckByJET"
        include("quality_tests/checkby_JET_tests.jl")
    end
end

@testset "Fit Tests" begin
    if GROUP == "RNODE"
        mts = Type{<:ContinuousNormalizingFlows.AbstractICNF}[RNODE]
        cmts = Type{<:ContinuousNormalizingFlows.AbstractCondICNF}[]
    elseif GROUP == "FFJORD"
        mts = Type{<:ContinuousNormalizingFlows.AbstractICNF}[FFJORD]
        cmts = Type{<:ContinuousNormalizingFlows.AbstractCondICNF}[]
    elseif GROUP == "Planar"
        mts = Type{<:ContinuousNormalizingFlows.AbstractICNF}[Planar]
        cmts = Type{<:ContinuousNormalizingFlows.AbstractCondICNF}[]
    elseif GROUP == "CondRNODE"
        mts = Type{<:ContinuousNormalizingFlows.AbstractICNF}[]
        cmts = Type{<:ContinuousNormalizingFlows.AbstractCondICNF}[CondRNODE]
    elseif GROUP == "CondFFJORD"
        mts = Type{<:ContinuousNormalizingFlows.AbstractICNF}[]
        cmts = Type{<:ContinuousNormalizingFlows.AbstractCondICNF}[CondFFJORD]
    elseif GROUP == "CondPlanar"
        mts = Type{<:ContinuousNormalizingFlows.AbstractICNF}[]
        cmts = Type{<:ContinuousNormalizingFlows.AbstractCondICNF}[CondPlanar]
    else
        mts = Type{<:ContinuousNormalizingFlows.AbstractICNF}[RNODE, FFJORD, Planar]
        cmts = Type{<:ContinuousNormalizingFlows.AbstractCondICNF}[
            CondRNODE,
            CondFFJORD,
            CondPlanar,
        ]
    end
    if GROUP == "All"
        nvars_ = Int[1, 2]
    else
        nvars_ = Int[1]
    end
    n_epochs = 2
    adtypes = ADTypes.AbstractADType[
        ADTypes.AutoZygote(),
        ADTypes.AutoReverseDiff(),
        ADTypes.AutoForwardDiff(),
    ]
    a_compute_modes = Type{<:ContinuousNormalizingFlows.ComputeMode}[
        ADVecJacVectorMode,
        # ADJacVecVectorMode,
        ZygoteVectorMode,
    ]
    m_compute_modes = Type{<:ContinuousNormalizingFlows.ComputeMode}[
        SDVecJacMatrixMode,
        # SDJacVecMatrixMode,
        ZygoteMatrixMode,
    ]
    data_types = Type{<:AbstractFloat}[Float32]
    resources = ComputationalResources.AbstractResource[ComputationalResources.CPU1()]
    if CUDA.has_cuda_gpu() && USE_GPU
        push!(resources, ComputationalResources.CUDALibs())
    end

    @testset "$resource | $data_type | $adtype | nvars = $nvars | $mt" for resource in
                                                                           resources,
        data_type in data_types,
        compute_mode in a_compute_modes,
        adtype in adtypes,
        nvars in nvars_,
        mt in mts

        data_dist =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
        r = convert.(data_type, rand(data_dist, nvars, 1))
        df = DataFrames.DataFrame(transpose(r), :auto)
        if mt <: Planar
            nn = PlanarLayer(nvars; use_bias = false)
        else
            nn = Lux.Dense(nvars => nvars; use_bias = false)
        end
        icnf = construct(mt, nn, nvars; data_type, compute_mode, resource)
        icnf.sol_kwargs[:sensealg] = SciMLSensitivity.ForwardDiffSensitivity()
        icnf.sol_kwargs[:verbose] = true
        model = ICNFModel(icnf; n_epochs, adtype)
        mach = MLJBase.machine(model, df)
        @test !isnothing(MLJBase.fit!(mach))
        @test !isnothing(MLJBase.transform(mach, df))
        @test !isnothing(MLJBase.fitted_params(mach))

        @test !isnothing(ICNFDist(mach, TrainMode()))
        @test !isnothing(ICNFDist(mach, TestMode()))
    end
    @testset "$resource | $data_type | $compute_mode | $adtype | nvars = $nvars | $mt" for resource in
                                                                                           resources,
        data_type in data_types,
        compute_mode in m_compute_modes,
        adtype in adtypes,
        nvars in nvars_,
        mt in mts

        data_dist =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
        r = convert.(data_type, rand(data_dist, nvars, 1))
        df = DataFrames.DataFrame(transpose(r), :auto)
        if mt <: Planar
            nn = PlanarLayer(nvars; use_bias = false)
        else
            nn = Lux.Dense(nvars => nvars; use_bias = false)
        end
        icnf = construct(mt, nn, nvars; data_type, compute_mode, resource)
        icnf.sol_kwargs[:sensealg] = SciMLSensitivity.ForwardDiffSensitivity()
        icnf.sol_kwargs[:verbose] = true
        model = ICNFModel(icnf; n_epochs, adtype)
        mach = MLJBase.machine(model, df)
        @test !isnothing(MLJBase.fit!(mach))
        @test !isnothing(MLJBase.transform(mach, df))
        @test !isnothing(MLJBase.fitted_params(mach))

        @test !isnothing(ICNFDist(mach, TrainMode()))
        @test !isnothing(ICNFDist(mach, TestMode()))
    end
    @testset "$resource | $data_type | $adtype | nvars = $nvars | $mt" for resource in
                                                                           resources,
        data_type in data_types,
        compute_mode in a_compute_modes,
        adtype in adtypes,
        nvars in nvars_,
        mt in cmts

        data_dist =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
        data_dist2 =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (4, 2))...)
        r = convert.(data_type, rand(data_dist, nvars, 1))
        r2 = convert.(data_type, rand(data_dist, nvars, 1))
        df = DataFrames.DataFrame(transpose(r), :auto)
        df2 = DataFrames.DataFrame(transpose(r2), :auto)
        if mt <: CondPlanar
            nn = PlanarLayer(nvars; use_bias = false, n_cond = nvars)
        else
            nn = Lux.Dense(2 * nvars => nvars; use_bias = false)
        end
        icnf = construct(mt, nn, nvars; data_type, compute_mode, resource)
        icnf.sol_kwargs[:sensealg] = SciMLSensitivity.ForwardDiffSensitivity()
        icnf.sol_kwargs[:verbose] = true
        model = CondICNFModel(icnf; n_epochs, adtype)
        mach = MLJBase.machine(model, (df, df2))
        @test !isnothing(MLJBase.fit!(mach))
        @test !isnothing(MLJBase.transform(mach, (df, df2)))
        @test !isnothing(MLJBase.fitted_params(mach))

        @test !isnothing(CondICNFDist(mach, TrainMode(), r2))
        @test !isnothing(CondICNFDist(mach, TestMode(), r2))
    end
    @testset "$resource | $data_type | $compute_mode | $adtype | nvars = $nvars | $mt" for resource in
                                                                                           resources,
        data_type in data_types,
        compute_mode in m_compute_modes,
        adtype in adtypes,
        nvars in nvars_,
        mt in cmts

        data_dist =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
        data_dist2 =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (4, 2))...)
        r = convert.(data_type, rand(data_dist, nvars, 1))
        r2 = convert.(data_type, rand(data_dist, nvars, 1))
        df = DataFrames.DataFrame(transpose(r), :auto)
        df2 = DataFrames.DataFrame(transpose(r2), :auto)
        if mt <: CondPlanar
            nn = PlanarLayer(nvars; use_bias = false, n_cond = nvars)
        else
            nn = Lux.Dense(2 * nvars => nvars; use_bias = false)
        end
        icnf = construct(mt, nn, nvars; data_type, compute_mode, resource)
        icnf.sol_kwargs[:sensealg] = SciMLSensitivity.ForwardDiffSensitivity()
        icnf.sol_kwargs[:verbose] = true
        model = CondICNFModel(icnf; n_epochs, adtype)
        mach = MLJBase.machine(model, (df, df2))
        @test !isnothing(MLJBase.fit!(mach))
        @test !isnothing(MLJBase.transform(mach, (df, df2)))
        @test !isnothing(MLJBase.fitted_params(mach))

        @test !isnothing(CondICNFDist(mach, TrainMode(), r2))
        @test !isnothing(CondICNFDist(mach, TestMode(), r2))
    end
end

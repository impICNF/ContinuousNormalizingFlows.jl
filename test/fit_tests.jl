@testset "Fit Tests" begin
    mts = if GROUP == "RNODE"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[RNODE]
    elseif GROUP == "FFJORD"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[FFJORD]
    elseif GROUP == "Planar"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[Planar]
    elseif GROUP == "CondRNODE"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[CondRNODE]
    elseif GROUP == "CondFFJORD"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[CondFFJORD]
    elseif GROUP == "CondPlanar"
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[CondPlanar]
    else
        Type{<:ContinuousNormalizingFlows.AbstractICNF}[
            RNODE,
            FFJORD,
            Planar,
            CondRNODE,
            CondFFJORD,
            CondPlanar,
        ]
    end
    n_epochs_ = Int[2]
    ndata_ = Int[4]
    nvars_ = Int[2]
    aug_steers = Bool[false, true]
    inplaces = Bool[false, true]
    adtypes = ADTypes.AbstractADType[
        ADTypes.AutoZygote(),
        ADTypes.AutoReverseDiff(),
        ADTypes.AutoForwardDiff(),
    ]
    compute_modes = Type{<:ContinuousNormalizingFlows.ComputeMode}[
        ADVecJacVectorMode,
        ADJacVecVectorMode,
        ZygoteVectorMode,
        SDVecJacMatrixMode,
        SDJacVecMatrixMode,
        ZygoteMatrixMode,
    ]
    data_types = Type{<:AbstractFloat}[Float32]
    resources = ComputationalResources.AbstractResource[ComputationalResources.CPU1()]
    if CUDA.has_cuda_gpu() && USE_GPU
        push!(resources, ComputationalResources.CUDALibs())
    end

    @testset "$resource | $data_type | $compute_mode | $adtype | inplace = $inplace | aug & steer = $aug_steer | nvars = $nvars | $mt" for resource in
                                                                                                                                           resources,
        data_type in data_types,
        compute_mode in compute_modes,
        adtype in adtypes,
        inplace in inplaces,
        aug_steer in aug_steers,
        nvars in nvars_,
        ndata in ndata_,
        n_epochs in n_epochs_,
        mt in mts

        data_dist =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
        data_dist2 =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (4, 2))...)
        r = convert.(data_type, rand(data_dist, nvars, ndata))
        r2 = convert.(data_type, rand(data_dist2, nvars, ndata))
        df = DataFrames.DataFrame(transpose(r), :auto)
        df2 = DataFrames.DataFrame(transpose(r2), :auto)

        nn = ifelse(
            mt <: Union{CondRNODE, CondFFJORD, CondPlanar},
            ifelse(
                mt <: CondPlanar,
                ifelse(
                    aug_steer,
                    PlanarLayer(nvars * 2, tanh; n_cond = nvars),
                    PlanarLayer(nvars, tanh; n_cond = nvars),
                ),
                ifelse(
                    aug_steer,
                    Lux.Dense(nvars * 3 => nvars * 2, tanh),
                    Lux.Dense(nvars * 2 => nvars, tanh),
                ),
            ),
            ifelse(
                mt <: Planar,
                ifelse(aug_steer, PlanarLayer(nvars * 2, tanh), PlanarLayer(nvars, tanh)),
                ifelse(
                    aug_steer,
                    Lux.Dense(nvars * 2 => nvars * 2, tanh),
                    Lux.Dense(nvars => nvars, tanh),
                ),
            ),
        )
        icnf = ifelse(
            aug_steer,
            construct(
                mt,
                nn,
                nvars,
                nvars;
                data_type,
                compute_mode,
                inplace,
                resource,
                steer_rate = convert(data_type, 1e-1),
                λ₃ = convert(data_type, 1e-2),
            ),
            construct(mt, nn, nvars; data_type, compute_mode, inplace, resource),
        )
        if mt <: Union{CondRNODE, CondFFJORD, CondPlanar}
            model = CondICNFModel(icnf; n_epochs, adtype)
            mach = MLJBase.machine(model, (df, df2))

            if (GROUP != "All") && (compute_mode <: SDJacVecMatrixMode || inplace)
                continue
            end

            @test !isnothing(MLJBase.fit!(mach))
            @test !isnothing(MLJBase.transform(mach, (df, df2)))
            @test !isnothing(MLJBase.fitted_params(mach))

            @test !isnothing(CondICNFDist(mach, TrainMode(), r2))
            @test !isnothing(CondICNFDist(mach, TestMode(), r2))
        else
            model = ICNFModel(icnf; n_epochs, adtype)
            mach = MLJBase.machine(model, df)

            if (GROUP != "All") && (compute_mode <: SDJacVecMatrixMode || inplace)
                continue
            end

            @test !isnothing(MLJBase.fit!(mach))
            @test !isnothing(MLJBase.transform(mach, df))
            @test !isnothing(MLJBase.fitted_params(mach))

            @test !isnothing(ICNFDist(mach, TrainMode()))
            @test !isnothing(ICNFDist(mach, TestMode()))
        end
    end
end

@testset "Fit Tests" begin
    mts = if GROUP == "RNODE"
        Type{<:ContinuousNormalizingFlows.AbstractFlows}[RNODE]
    elseif GROUP == "FFJORD"
        Type{<:ContinuousNormalizingFlows.AbstractFlows}[FFJORD]
    elseif GROUP == "Planar"
        Type{<:ContinuousNormalizingFlows.AbstractFlows}[Planar]
    elseif GROUP == "CondRNODE"
        Type{<:ContinuousNormalizingFlows.AbstractFlows}[CondRNODE]
    elseif GROUP == "CondFFJORD"
        Type{<:ContinuousNormalizingFlows.AbstractFlows}[CondFFJORD]
    elseif GROUP == "CondPlanar"
        Type{<:ContinuousNormalizingFlows.AbstractFlows}[CondPlanar]
    else
        Type{<:ContinuousNormalizingFlows.AbstractFlows}[
            RNODE,
            FFJORD,
            Planar,
            CondRNODE,
            CondFFJORD,
            CondPlanar,
        ]
    end
    nvars_ = ifelse(GROUP == "All", Int[1, 2], Int[1])
    inplaces = ifelse(GROUP == "All", Bool[false, true], Bool[false])
    aug_steers = ifelse(GROUP == "All", Bool[false, true], Bool[true])
    n_epochs = 2
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
        # SDJacVecMatrixMode,
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
        mt in mts

        data_dist =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
        r = convert.(data_type, rand(data_dist, nvars, 1))
        df = DataFrames.DataFrame(transpose(r), :auto)
        data_dist2 =
            Distributions.Beta{data_type}(convert(Tuple{data_type, data_type}, (4, 2))...)
        r2 = convert.(data_type, rand(data_dist, nvars, 1))
        df2 = DataFrames.DataFrame(transpose(r2), :auto)

        nn = ifelse(
            mt <: ContinuousNormalizingFlows.AbstractCondICNF,
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
                steer_rate = convert(data_type, 0.1),
                sol_kwargs = merge(
                    ContinuousNormalizingFlows.sol_kwargs_medium,
                    (
                        sensealg = InterpolatingAdjoint(;
                            autodiff = true,
                            autojacvec = true,
                            checkpointing = true,
                        )
                    ),
                ),
            ),
            construct(
                mt,
                nn,
                nvars;
                data_type,
                compute_mode,
                inplace,
                resource,
                sol_kwargs = merge(
                    ContinuousNormalizingFlows.sol_kwargs_medium,
                    (
                        sensealg = InterpolatingAdjoint(;
                            autodiff = true,
                            autojacvec = true,
                            checkpointing = true,
                        )
                    ),
                ),
            ),
        )
        if mt <: ContinuousNormalizingFlows.AbstractCondICNF
            model = CondICNFModel(icnf; n_epochs, adtype)
            mach = MLJBase.machine(model, (df, df2))
            @test !isnothing(MLJBase.fit!(mach))
            @test !isnothing(MLJBase.transform(mach, (df, df2)))
            @test !isnothing(MLJBase.fitted_params(mach))

            @test !isnothing(CondICNFDist(mach, TrainMode(), r2))
            @test !isnothing(CondICNFDist(mach, TestMode(), r2))
        else
            model = ICNFModel(icnf; n_epochs, adtype)
            mach = MLJBase.machine(model, df)
            @test !isnothing(MLJBase.fit!(mach))
            @test !isnothing(MLJBase.transform(mach, df))
            @test !isnothing(MLJBase.fitted_params(mach))

            @test !isnothing(ICNFDist(mach, TrainMode()))
            @test !isnothing(ICNFDist(mach, TestMode()))
        end
    end
end

@testset "Fit Tests" begin
    mts =
        SMALL ? Type{<:ICNF.AbstractICNF}[RNODE] :
        Type{<:ICNF.AbstractICNF}[RNODE, FFJORD, Planar]
    cmts =
        SMALL ? Type{<:ICNF.AbstractCondICNF}[CondRNODE] :
        Type{<:ICNF.AbstractCondICNF}[CondRNODE, CondFFJORD, CondPlanar]
    ats = Type{<:AbstractArray}[Array]
    if CUDA.has_cuda_gpu() && !SMALL
        push!(ats, CUDA.CuArray)
    end
    tps = Type{<:AbstractFloat}[Float32]
    cmodes =
        Type{<:ICNF.ComputeMode}[ZygoteMatrixMode, SDVecJacMatrixMode, SDJacVecMatrixMode]
    nvars_ = (1:2)
    adb_list = AbstractDifferentiation.AbstractBackend[
        AbstractDifferentiation.ZygoteBackend(),
        AbstractDifferentiation.ReverseDiffBackend(),
        AbstractDifferentiation.ForwardDiffBackend(),
        AbstractDifferentiation.TrackerBackend(),
        AbstractDifferentiation.FiniteDifferencesBackend(),
    ]
    go_ads = SciMLBase.AbstractADType[
        Optimization.AutoZygote(),
        Optimization.AutoReverseDiff(),
        Optimization.AutoForwardDiff(),
        Optimization.AutoTracker(),
        Optimization.AutoFiniteDiff(),
    ]

    @testset "$at | $tp | $(typeof(adb_u).name.name) for internal | $(typeof(go_ad).name.name) for fitting | $nvars Vars | $mt" for at in
                                                                                                                                    ats,
        tp in tps,
        adb_u in adb_list,
        go_ad in go_ads,
        nvars in nvars_,
        mt in mts

        adb_u isa AbstractDifferentiation.FiniteDifferencesBackend && continue
        adb_u isa AbstractDifferentiation.ReverseDiffBackend && continue
        adb_u isa AbstractDifferentiation.TrackerBackend && mt <: Planar && continue
        go_ad isa Optimization.AutoTracker && continue

        data_dist = Distributions.Beta{tp}(convert(Tuple{tp, tp}, (2, 4))...)
        r = convert(at{tp}, rand(data_dist, nvars, 2))
        df = DataFrames.DataFrame(transpose(r), :auto)
        if mt <: Planar
            nn = PlanarLayer(nvars, tanh)
        else
            nn = Lux.Dense(nvars => nvars, tanh)
        end
        icnf = construct(
            mt,
            nn,
            nvars;
            data_type = tp,
            array_type = at,
            differentiation_backend = adb_u,
        )
        model = ICNFModel(
            icnf;
            n_epochs = 2,
            adtype = go_ad,
            resource = (at == CUDA.CuArray) ? ComputationalResources.CUDALibs() :
                       ComputationalResources.CPU1(),
        )
        mach = MLJBase.machine(model, df)
        @test !isnothing(MLJBase.fit!(mach))
        @test !isnothing(MLJBase.transform(mach, df))
        @test !isnothing(MLJBase.fitted_params(mach))
    end
    @testset "$at | $tp | $cmode | $(typeof(go_ad).name.name) for fitting | $nvars Vars | $mt" for at in
                                                                                                   ats,
        tp in tps,
        cmode in cmodes,
        go_ad in go_ads,
        nvars in nvars_,
        mt in mts

        cmode <: SDJacVecMatrixMode && continue
        go_ad isa Optimization.AutoTracker && continue

        data_dist = Distributions.Beta{tp}(convert(Tuple{tp, tp}, (2, 4))...)
        r = convert(at{tp}, rand(data_dist, nvars, 2))
        df = DataFrames.DataFrame(transpose(r), :auto)
        if mt <: Planar
            nn = PlanarLayer(nvars, tanh)
        else
            nn = Lux.Dense(nvars => nvars, tanh)
        end
        icnf =
            construct(mt, nn, nvars; data_type = tp, array_type = at, compute_mode = cmode)
        model = ICNFModel(
            icnf;
            n_epochs = 2,
            adtype = go_ad,
            resource = (at == CUDA.CuArray) ? ComputationalResources.CUDALibs() :
                       ComputationalResources.CPU1(),
        )
        mach = MLJBase.machine(model, df)
        @test !isnothing(MLJBase.fit!(mach))
        @test !isnothing(MLJBase.transform(mach, df))
        @test !isnothing(MLJBase.fitted_params(mach))
    end
    @testset "$at | $tp | $(typeof(adb_u).name.name) for internal | $(typeof(go_ad).name.name) for fitting | $nvars Vars | $mt" for at in
                                                                                                                                    ats,
        tp in tps,
        adb_u in adb_list,
        go_ad in go_ads,
        nvars in nvars_,
        mt in cmts

        adb_u isa AbstractDifferentiation.FiniteDifferencesBackend && continue
        adb_u isa AbstractDifferentiation.ReverseDiffBackend && continue
        adb_u isa AbstractDifferentiation.TrackerBackend && continue
        adb_u isa AbstractDifferentiation.TrackerBackend && mt <: CondPlanar && continue
        go_ad isa Optimization.AutoTracker && continue

        data_dist = Distributions.Beta{tp}(convert(Tuple{tp, tp}, (2, 4))...)
        data_dist2 = Distributions.Beta{tp}(convert(Tuple{tp, tp}, (4, 2))...)
        r = convert(at{tp}, rand(data_dist, nvars, 2))
        r2 = convert(at{tp}, rand(data_dist, nvars, 2))
        df = DataFrames.DataFrame(transpose(r), :auto)
        df2 = DataFrames.DataFrame(transpose(r2), :auto)
        if mt <: CondPlanar
            nn = PlanarLayer(nvars, tanh; cond = true)
        else
            nn = Lux.Dense(2 * nvars => nvars, tanh)
        end
        icnf = construct(
            mt,
            nn,
            nvars;
            data_type = tp,
            array_type = at,
            differentiation_backend = adb_u,
        )
        model = CondICNFModel(
            icnf;
            n_epochs = 2,
            adtype = go_ad,
            resource = (at == CUDA.CuArray) ? ComputationalResources.CUDALibs() :
                       ComputationalResources.CPU1(),
        )
        mach = MLJBase.machine(model, (df, df2))
        @test !isnothing(MLJBase.fit!(mach))
        @test !isnothing(MLJBase.transform(mach, (df, df2)))
        @test !isnothing(MLJBase.fitted_params(mach))
    end
    @testset "$at | $tp | $cmode | $(typeof(go_ad).name.name) for fitting | $nvars Vars | $mt" for at in
                                                                                                   ats,
        tp in tps,
        cmode in cmodes,
        go_ad in go_ads,
        nvars in nvars_,
        mt in cmts

        cmode <: SDJacVecMatrixMode && continue
        go_ad isa Optimization.AutoTracker && continue

        data_dist = Distributions.Beta{tp}(convert(Tuple{tp, tp}, (2, 4))...)
        data_dist2 = Distributions.Beta{tp}(convert(Tuple{tp, tp}, (4, 2))...)
        r = convert(at{tp}, rand(data_dist, nvars, 2))
        r2 = convert(at{tp}, rand(data_dist, nvars, 2))
        df = DataFrames.DataFrame(transpose(r), :auto)
        df2 = DataFrames.DataFrame(transpose(r2), :auto)
        if mt <: CondPlanar
            nn = PlanarLayer(nvars, tanh; cond = true)
        else
            nn = Lux.Dense(2 * nvars => nvars, tanh)
        end
        icnf =
            construct(mt, nn, nvars; data_type = tp, array_type = at, compute_mode = cmode)
        model = CondICNFModel(
            icnf;
            n_epochs = 2,
            adtype = go_ad,
            resource = (at == CUDA.CuArray) ? ComputationalResources.CUDALibs() :
                       ComputationalResources.CPU1(),
        )
        mach = MLJBase.machine(model, (df, df2))
        @test !isnothing(MLJBase.fit!(mach))
        @test !isnothing(MLJBase.transform(mach, (df, df2)))
        @test !isnothing(MLJBase.fitted_params(mach))
    end
end

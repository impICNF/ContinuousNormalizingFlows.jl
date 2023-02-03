@testset "Fit Tests" begin
    mts =
        SMALL ? Type{<:ICNF.AbstractICNF}[RNODE] :
        Type{<:ICNF.AbstractICNF}[RNODE, FFJORD, Planar]
    cmts =
        SMALL ? Type{<:ICNF.AbstractCondICNF}[CondRNODE] :
        Type{<:ICNF.AbstractCondICNF}[CondRNODE, CondFFJORD, CondPlanar]
    ats = Type{<:AbstractArray}[Array]
    if has_cuda_gpu() && !SMALL
        push!(ats, CuArray)
    end
    tps =
        SMALL ? Type{<:AbstractFloat}[Float32] :
        Type{<:AbstractFloat}[Float64, Float32, Float16]
    nvars_ = (1:2)
    go_ads = SciMLBase.AbstractADType[
        Optimization.AutoZygote(),
        Optimization.AutoReverseDiff(),
        Optimization.AutoForwardDiff(),
        Optimization.AutoTracker(),
        Optimization.AutoFiniteDiff(),
    ]

    @testset "$at | $tp | $nvars Vars | $mt" for at in ats,
        tp in tps,
        nvars in nvars_,
        mt in mts

        data_dist = Beta{tp}(convert(Tuple{tp, tp}, (2, 4))...)
        r = convert(Matrix{tp}, rand(data_dist, nvars, 2))
        df = DataFrame(transpose(r), :auto)

        @testset "Using $(typeof(go_ad).name.name)" for go_ad in go_ads
            if mt <: Planar
                nn = PlanarLayer(nvars, tanh)
            else
                nn = Dense(nvars => nvars, tanh)
            end
            icnf = mt{tp, at}(nn, nvars)
            model = ICNFModel(
                icnf;
                n_epochs = 2,
                adtype = go_ad,
                resource = (at == CuArray) ? CUDALibs() : CPU1(),
            )
            mach = machine(model, df)
            @test !isnothing(fit!(mach))
            @test !isnothing(MLJBase.transform(mach, df))
            @test !isnothing(MLJBase.fitted_params(mach))
        end
    end
    @testset "$at | $tp | $nvars Vars | $mt" for at in ats,
        tp in tps,
        nvars in nvars_,
        mt in cmts

        data_dist = Beta{tp}(convert(Tuple{tp, tp}, (2, 4))...)
        data_dist2 = Beta{tp}(convert(Tuple{tp, tp}, (4, 2))...)
        r = convert(Matrix{tp}, rand(data_dist, nvars, 2))
        r2 = convert(Matrix{tp}, rand(data_dist, nvars, 2))
        df = DataFrame(transpose(r), :auto)
        df2 = DataFrame(transpose(r2), :auto)

        @testset "Using $(typeof(go_ad).name.name)" for go_ad in go_ads
            if mt <: CondPlanar
                nn = PlanarLayer(nvars, tanh; cond = true)
            else
                nn = Dense(2 * nvars => nvars, tanh)
            end
            icnf = mt{tp, at}(nn, nvars)
            model = CondICNFModel(
                icnf;
                n_epochs = 2,
                adtype = go_ad,
                resource = (at == CuArray) ? CUDALibs() : CPU1(),
            )
            mach = machine(model, (df, df2))
            @test !isnothing(fit!(mach))
            @test !isnothing(MLJBase.transform(mach, (df, df2)))
            @test !isnothing(MLJBase.fitted_params(mach))
        end
    end
end

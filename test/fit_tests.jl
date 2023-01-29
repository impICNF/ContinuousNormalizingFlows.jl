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
    tps = Type{<:AbstractFloat}[Float64, Float32, Float16]
    nvars_ = (1:2)
    go_ads = SciMLBase.AbstractADType[
        Optimization.AutoZygote(),
        Optimization.AutoReverseDiff(),
        Optimization.AutoForwardDiff(),
        Optimization.AutoTracker(),
        Optimization.AutoFiniteDiff(),
    ]
    n_epochs = 2
    batch_size = 2
    n_batch = 2
    n = n_batch * batch_size

    @testset "$at | $tp | $nvars Vars | $mt" for at in ats,
        tp in tps,
        nvars in nvars_,
        mt in mts

        data_dist = Beta{tp}(convert(Tuple{tp, tp}, (2, 4))...)
        r = convert(Matrix{tp}, rand(data_dist, nvars, n))
        df = DataFrame(transpose(r), :auto)

        @testset "Using $(typeof(go_ad).name.name)" for go_ad in go_ads
            if mt <: Planar
                nn = PlanarNN(nvars, tanh)
            else
                nn = Chain(Dense(nvars => nvars, tanh))
            end
            icnf = mt{tp, at}(nn, nvars)
            model = ICNFModel(icnf; n_epochs, batch_size, adtype = go_ad)
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
        r = convert(Matrix{tp}, rand(data_dist, nvars, n))
        r2 = convert(Matrix{tp}, rand(data_dist, nvars, n))
        df = DataFrame(transpose(r), :auto)
        df2 = DataFrame(transpose(r2), :auto)

        @testset "Using $(typeof(go_ad).name.name)" for go_ad in go_ads
            if mt <: CondPlanar
                nn = PlanarNN(nvars, tanh; cond = true)
            else
                nn = Chain(Dense(2 * nvars => nvars, tanh))
            end
            icnf = mt{tp, at}(nn, nvars)
            model = CondICNFModel(icnf; n_epochs, batch_size, adtype = go_ad)
            mach = machine(model, (df, df2))
            @test !isnothing(fit!(mach))
            @test !isnothing(MLJBase.transform(mach, (df, df2)))
            @test !isnothing(MLJBase.fitted_params(mach))
        end
    end
end

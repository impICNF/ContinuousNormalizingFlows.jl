@testset "CondFFJORD" begin
    @testset "$cr-$tp-$nvars-Var-Smoke-Test" for
            cr in [CPU1(), CUDALibs()],
            tp in [Float64, Float32, Float16],
            nvars in 1:3
        icnf = CondFFJORD{tp}(Dense(nvars*2, nvars), nvars; acceleration=cr)
        ufd = copy(icnf.p)
        n = 8
        r = rand(tp, nvars, n)
        r2 = rand(tp, nvars, n)

        @test !isnothing(inference(icnf, TestMode(), r, r2))
        @test !isnothing(inference(icnf, TrainMode(), r, r2))

        @test !isnothing(generate(icnf, TestMode(), r2, n))
        @test !isnothing(generate(icnf, TrainMode(), r2, n))

        @test !isnothing(icnf(r, r2))
        @test !isnothing(loss_f(icnf)(r, r2))

        let
            model=(
                m=icnf,
                loss=loss_f(icnf),
                optimizer=AMSGrad(),
                n_epochs=8,
                batch_size=128,
                cb_timeout=16,
            )
            data = [(r, r2),]
            Flux.Optimise.@epochs model.n_epochs Flux.Optimise.train!(model.loss, Flux.params(model.m), data, model.optimizer; cb=Flux.throttle(cb_f(model.m, model.loss, data), model.cb_timeout))
        end
        fd = icnf.p
        if tp <: Float16
            @test_broken fd != ufd
        else
            @test fd != ufd
        end
    end
end

default_tspan = (0, 1)
default_solvealg = Tsit5(; thread = OrdinaryDiffEq.True())
default_sensealg = InterpolatingAdjoint(;
    autodiff = true,
    checkpointing = false,
    noisemixing = false,
    chunk_size = 0,
    autojacvec = ZygoteVJP(),
)
default_optimizer = Dict(
    FluxOptApp => Flux.AMSGrad(0.001, (0.9, 0.999), eps()),
    OptimOptApp => BFGS(
        alphaguess = InitialHagerZhang(),
        linesearch = HagerZhang(),
        manifold = Flat(),
    ),
    SciMLOptApp => Optimisers.AMSGrad(0.001, (0.9, 0.999), eps()),
)

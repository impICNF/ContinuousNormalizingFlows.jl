default_tspan = (0, 1)
default_optimizer = Dict(
    FluxOptApp => Flux.AMSGrad(0.001, (0.9, 0.999), eps()),
    OptimOptApp => BFGS(;
        alphaguess = InitialHagerZhang(),
        linesearch = HagerZhang(),
        manifold = Flat(),
    ),
    SciMLOptApp => Optimisers.AMSGrad(0.001, (0.9, 0.999), eps()),
)

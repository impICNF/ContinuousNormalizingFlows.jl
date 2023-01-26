default_tspan = (0, 1)
default_optimizer = Optimisers.OptimiserChain(
    Optimisers.WeightDecay(5e-4),
    Optimisers.AMSGrad(1e-3, (9e-1, 9.99e-1), eps(Float64)),
)

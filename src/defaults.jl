default_tspan = (0, 1)
default_optimizer = Optimisers.OptimiserChain(
    Optimisers.WeightDecay(),
    Optimisers.AMSGrad(),
)

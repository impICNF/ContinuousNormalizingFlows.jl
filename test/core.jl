function gen_dist(scl = 5, nvars = 2, nmix = nv)
    Distributions.Product([
        Distributions.MixtureModel([Distributions.Normal(scl * rand(), scl * rand()) for _ in 1:nmix]) for _ in 1:nvars
    ])
end

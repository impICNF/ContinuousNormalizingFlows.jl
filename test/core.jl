function gen_dist(scl = 5, nvars = 2, nmix = nv)
    Product([
        MixtureModel([Normal(scl * rand(), scl * rand()) for _ = 1:nmix]) for _ = 1:nvars
    ])
end

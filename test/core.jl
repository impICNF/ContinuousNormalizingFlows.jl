function gen_dist(scl=5, nvars=2, nmix=nv)
    Product([MixtureModel([Normal(scl*rand(), scl*rand()) for _ in 1:nmix]) for _ in 1:nvars])
end

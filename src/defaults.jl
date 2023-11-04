sol_kwargs_high = (
    alg_hints = [:nonstiff],
    alg = VCABM(),
    sensealg = QuadratureAdjoint(; autodiff = true, autojacvec = ZygoteVJP()),
    reltol = sqrt(eps(one(Float32))),
    abstol = eps(one(Float32)),
    maxiters = typemax(Int32),
)
sol_kwargs_medium = (
    alg_hints = [:nonstiff, :memorybound],
    save_everystep = false,
    alg = VCABM(),
    sensealg = InterpolatingAdjoint(;
        autodiff = true,
        autojacvec = ZygoteVJP(),
        checkpointing = true,
    ),
    reltol = sqrt(eps(one(Float32))),
    abstol = eps(one(Float32)),
    maxiters = typemax(Int32),
)
sol_kwargs_low = (
    alg_hints = [:nonstiff, :memorybound],
    save_everystep = false,
    alg = VCABM(),
    sensealg = InterpolatingAdjoint(;
        autodiff = true,
        autojacvec = ZygoteVJP(),
        checkpointing = true,
    ),
    reltol = sqrt(sqrt(eps(one(Float32)))),
    abstol = eps(one(Float32)),
    maxiters = typemax(Int32),
)

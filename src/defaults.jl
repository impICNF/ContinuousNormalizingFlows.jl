const sol_kwargs_defaults = (
    high = (
        alg_hints = [:nonstiff],
        alg = VCABM(),
        sensealg = QuadratureAdjoint(; autodiff = true, autojacvec = ZygoteVJP()),
        reltol = eps(one(Float32)),
        abstol = eps(one(Float32)) * eps(one(Float32)),
        maxiters = typemax(Int32),
    ),
    medium = (
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
    ),
    low = (
        alg_hints = [:nonstiff, :memorybound],
        save_everystep = false,
        alg = VCABM(),
        sensealg = InterpolatingAdjoint(;
            autodiff = true,
            autojacvec = ZygoteVJP(),
            checkpointing = true,
        ),
        reltol = sqrt(sqrt(eps(one(Float32)))),
        abstol = sqrt(eps(one(Float32))),
        maxiters = typemax(Int32),
    ),
)

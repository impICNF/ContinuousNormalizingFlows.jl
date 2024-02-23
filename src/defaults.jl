const sol_kwargs_defaults = (
    high = (
        alg = Vern6(; thread = OrdinaryDiffEq.True(), lazy = true),
        sensealg = QuadratureAdjoint(; autodiff = true, autojacvec = ZygoteVJP()),
        reltol = eps(one(Float32)),
        abstol = eps(one(Float32)) * eps(one(Float32)),
        maxiters = typemax(Int32),
    ),
    medium = (
        save_everystep = false,
        alg = Vern6(; thread = OrdinaryDiffEq.True(), lazy = true),
        sensealg = InterpolatingAdjoint(;
            autodiff = true,
            autojacvec = ZygoteVJP(),
            checkpointing = true,
        ),
        reltol = sqrt(eps(one(Float32))),
        abstol = eps(one(Float32)),
        maxiters = typemax(Int32),
    ),
    medium_noad = (
        save_everystep = false,
        alg = Vern6(; thread = OrdinaryDiffEq.True(), lazy = true),
        reltol = sqrt(eps(one(Float32))),
        abstol = eps(one(Float32)),
        maxiters = typemax(Int32),
    ),
    low = (
        save_everystep = false,
        alg = Vern6(; thread = OrdinaryDiffEq.True(), lazy = true),
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

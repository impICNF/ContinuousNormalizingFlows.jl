export inference, generate, loss

function inference_prob(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steer_rate::AbstractFloat = steer_rate_value(icnf),
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    zrs = zeros_T_AT(resource, icnf, n_aug_input + n_aug + 1)
    func = ODEFunction{false, SciMLBase.FullSpecialize}(
        (u, p, t) -> augmented_f(
            u,
            p,
            t,
            icnf,
            mode,
            st;
            resource,
            differentiation_backend,
            rng,
        ),
    )
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        vcat(xs, zrs),
        steer_tspan(icnf, mode, tspan, steer_rate, rng),
        ps,
        sol_args...;
        sol_kwargs...,
    )
    prob
end

function inference(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steer_rate::AbstractFloat = steer_rate_value(icnf),
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    prob = inference_prob(
        icnf,
        mode,
        xs,
        ps,
        st;
        tspan,
        steer_rate,
        basedist,
        differentiation_backend,
        rng,
        sol_args,
        sol_kwargs,
    )
    n_aug = n_augment(icnf, mode)
    sol = solve(prob, sol_args...; sol_kwargs...)
    fsol = @view sol[:, end]
    z = @view fsol[begin:(end - n_aug - 1)]
    Δlogp = fsol[(end - n_aug)]
    augs = @view fsol[(end - n_aug + 1):end]
    logp̂x = logpdf(basedist, z) - Δlogp
    iszero(n_aug) ? (logp̂x,) : (logp̂x, augs...)
end

function inference_prob(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steer_rate::AbstractFloat = steer_rate_value(icnf),
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    zrs = zeros_T_AT(resource, icnf, n_aug_input + n_aug + 1, size(xs, 2))
    func = ODEFunction{false, SciMLBase.FullSpecialize}(
        (u, p, t) -> augmented_f(
            u,
            p,
            t,
            icnf,
            mode,
            st;
            resource,
            differentiation_backend,
            rng,
        ),
    )
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        vcat(xs, zrs),
        steer_tspan(icnf, mode, tspan, steer_rate, rng),
        ps,
        sol_args...;
        sol_kwargs...,
    )
    prob
end

function inference(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steer_rate::AbstractFloat = steer_rate_value(icnf),
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    prob = inference_prob(
        icnf,
        mode,
        xs,
        ps,
        st;
        tspan,
        steer_rate,
        basedist,
        differentiation_backend,
        rng,
        sol_args,
        sol_kwargs,
    )
    n_aug = n_augment(icnf, mode)
    sol = solve(prob, sol_args...; sol_kwargs...)
    fsol = @view sol[:, :, end]
    z = @view fsol[begin:(end - n_aug - 1), :]
    Δlogp = @view fsol[(end - n_aug), :]
    augs = @view fsol[(end - n_aug + 1):end, :]
    logp̂x = logpdf(basedist, z) - Δlogp
    iszero(n_aug) ? (logp̂x,) : (logp̂x, eachrow(augs)...)
end

function generate_prob(
    icnf::AbstractICNF{T, <:VectorMode},
    mode::Mode,
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steer_rate::AbstractFloat = steer_rate_value(icnf),
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    new_xs = rand_cstm_AT(resource, icnf, basedist, rng)
    zrs = zeros_T_AT(resource, icnf, n_aug + 1)
    func = ODEFunction{false, SciMLBase.FullSpecialize}(
        (u, p, t) -> augmented_f(
            u,
            p,
            t,
            icnf,
            mode,
            st;
            resource,
            differentiation_backend,
            rng,
        ),
    )
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        vcat(new_xs, zrs),
        reverse(steer_tspan(icnf, mode, tspan, steer_rate, rng)),
        ps,
        sol_args...;
        sol_kwargs...,
    )
    prob
end
function generate(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steer_rate::AbstractFloat = steer_rate_value(icnf),
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    prob = generate_prob(
        icnf,
        mode,
        ps,
        st;
        tspan,
        steer_rate,
        basedist,
        differentiation_backend,
        rng,
        sol_args,
        sol_kwargs,
    )
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    sol = solve(prob, sol_args...; sol_kwargs...)
    fsol = @view sol[:, end]
    z = @view fsol[begin:(end - n_aug_input - n_aug - 1)]
    z
end

function generate_prob(
    icnf::AbstractICNF{T, <:MatrixMode},
    mode::Mode,
    ps::Any,
    st::Any,
    n::Integer;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steer_rate::AbstractFloat = steer_rate_value(icnf),
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    new_xs = rand_cstm_AT(resource, icnf, basedist, rng, n)
    zrs = zeros_T_AT(resource, icnf, n_aug + 1, size(new_xs, 2))
    func = ODEFunction{false, SciMLBase.FullSpecialize}(
        (u, p, t) -> augmented_f(
            u,
            p,
            t,
            icnf,
            mode,
            st;
            resource,
            differentiation_backend,
            rng,
        ),
    )
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        vcat(new_xs, zrs),
        reverse(steer_tspan(icnf, mode, tspan, steer_rate, rng)),
        ps,
        sol_args...;
        sol_kwargs...,
    )
    prob
end

function generate(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    ps::Any,
    st::Any,
    n::Integer;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steer_rate::AbstractFloat = steer_rate_value(icnf),
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    prob = generate_prob(
        icnf,
        mode,
        ps,
        st,
        n;
        tspan,
        steer_rate,
        basedist,
        differentiation_backend,
        rng,
        sol_args,
        sol_kwargs,
    )
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    sol = solve(prob, sol_args...; sol_kwargs...)
    fsol = @view sol[:, :, end]
    z = @view fsol[begin:(end - n_aug_input - n_aug - 1), :]
    z
end

@inline function loss(
    icnf::AbstractICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steer_rate::AbstractFloat = steer_rate_value(icnf),
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    -first(
        inference(
            icnf,
            mode,
            xs,
            ps,
            st;
            tspan,
            basedist,
            differentiation_backend,
            rng,
            sol_args,
            sol_kwargs,
        ),
    )
end

@inline function loss(
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steer_rate::AbstractFloat = steer_rate_value(icnf),
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    -mean(
        first(
            inference(
                icnf,
                mode,
                xs,
                ps,
                st;
                tspan,
                basedist,
                differentiation_backend,
                rng,
                sol_args,
                sol_kwargs,
            ),
        ),
    )
end

function augmented_f(
    u,
    p,
    t,
    icnf::AbstractICNF{<:AbstractFloat, <:ADVectorMode},
    mode::TestMode,
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = @view u[begin:(end - n_aug - 1)]
    mz, J = AbstractDifferentiation.value_and_jacobian(
        differentiation_backend,
        x -> first(LuxCore.apply(icnf.nn, x, p, st)),
        z,
    )
    trace_J = tr(only(J))
    vcat(mz, -trace_J)
end

function augmented_f(
    u,
    p,
    t,
    icnf::AbstractICNF{<:AbstractFloat, <:MatrixMode},
    mode::TestMode,
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    z = @view u[begin:(end - n_aug - 1), :]
    mz, J =
        jacobian_batched(icnf, x -> first(LuxCore.apply(icnf.nn, x, p, st)), z; resource)
    trace_J = transpose(tr.(eachslice(J; dims = 3)))
    vcat(mz, -trace_J)
end

@inline function (icnf::AbstractICNF)(xs::Any, ps::Any, st::Any)
    first(inference(icnf, TrainMode(), xs, ps, st))
end

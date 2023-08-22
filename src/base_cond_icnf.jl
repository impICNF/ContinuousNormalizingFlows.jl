export inference, generate, loss

function inference_prob(
    icnf::AbstractCondICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    zrs = zeros_T_AT(resource, icnf, n_aug_input + n_aug + 1)
    ϵ = randn_T_AT(resource, icnf, rng, icnf.nvars + n_aug_input)
    func = ODEFunction{false, SciMLBase.FullSpecialize}(
        (u, p, t) -> augmented_f(
            u,
            p,
            t,
            icnf,
            mode,
            ys,
            ϵ,
            st;
            resource,
            differentiation_backend,
            rng,
        ),
    )
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        vcat(xs, zrs),
        steer_tspan(icnf, mode; tspan, steerdist, rng),
        ps,
        sol_args...;
        sol_kwargs...,
    )
    prob
end

function inference(
    icnf::AbstractCondICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
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
        ys,
        ps,
        st;
        tspan,
        steerdist,
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
    logp̂x = logpdf(basedist, z) - Δlogp
    if iszero(n_aug)
        (logp̂x,)
    else
        augs = @view fsol[(end - n_aug + 1):end]
        (logp̂x, augs...)
    end
end

function inference_prob(
    icnf::AbstractCondICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    zrs = zeros_T_AT(resource, icnf, n_aug_input + n_aug + 1, size(xs, 2))
    ϵ = randn_T_AT(resource, icnf, rng, icnf.nvars + n_aug_input, size(xs, 2))
    func = ODEFunction{false, SciMLBase.FullSpecialize}(
        (u, p, t) -> augmented_f(
            u,
            p,
            t,
            icnf,
            mode,
            ys,
            ϵ,
            st;
            resource,
            differentiation_backend,
            rng,
        ),
    )
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        vcat(xs, zrs),
        steer_tspan(icnf, mode; tspan, steerdist, rng),
        ps,
        sol_args...;
        sol_kwargs...,
    )
    prob
end

function inference(
    icnf::AbstractCondICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
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
        ys,
        ps,
        st;
        tspan,
        steerdist,
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
    logp̂x = logpdf(basedist, z) - Δlogp
    if iszero(n_aug)
        (logp̂x,)
    else
        augs = @view fsol[(end - n_aug + 1):end, :]
        (logp̂x, eachrow(augs)...)
    end
end

function generate_prob(
    icnf::AbstractCondICNF{T, <:VectorMode},
    mode::Mode,
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    new_xs = rand_cstm_AT(resource, icnf, basedist, rng)
    zrs = zeros_T_AT(resource, icnf, n_aug + 1)
    ϵ = randn_T_AT(resource, icnf, rng, icnf.nvars + n_aug_input)
    func = ODEFunction{false, SciMLBase.FullSpecialize}(
        (u, p, t) -> augmented_f(
            u,
            p,
            t,
            icnf,
            mode,
            ys,
            ϵ,
            st;
            resource,
            differentiation_backend,
            rng,
        ),
    )
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        vcat(new_xs, zrs),
        reverse(steer_tspan(icnf, mode; tspan, steerdist, rng)),
        ps,
        sol_args...;
        sol_kwargs...,
    )
    prob
end

function generate(
    icnf::AbstractCondICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    prob = generate_prob(
        icnf,
        mode,
        ys,
        ps,
        st;
        tspan,
        steerdist,
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
    icnf::AbstractCondICNF{T, <:MatrixMode},
    mode::Mode,
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::Any,
    n::Int;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
) where {T <: AbstractFloat}
    n_aug = n_augment(icnf, mode)
    n_aug_input = n_augment_input(icnf)
    new_xs = rand_cstm_AT(resource, icnf, basedist, rng, n)
    zrs = zeros_T_AT(resource, icnf, n_aug + 1, size(new_xs, 2))
    ϵ = randn_T_AT(resource, icnf, rng, icnf.nvars + n_aug_input, size(new_xs, 2))
    func = ODEFunction{false, SciMLBase.FullSpecialize}(
        (u, p, t) -> augmented_f(
            u,
            p,
            t,
            icnf,
            mode,
            ys,
            ϵ,
            st;
            resource,
            differentiation_backend,
            rng,
        ),
    )
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        func,
        vcat(new_xs, zrs),
        reverse(steer_tspan(icnf, mode; tspan, steerdist, rng)),
        ps,
        sol_args...;
        sol_kwargs...,
    )
    prob
end

function generate(
    icnf::AbstractCondICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::Any,
    n::Int;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
    basedist::Distribution = icnf.basedist,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
    sol_args::Tuple = icnf.sol_args,
    sol_kwargs::Dict = icnf.sol_kwargs,
)
    prob = generate_prob(
        icnf,
        mode,
        ys,
        ps,
        st,
        n;
        tspan,
        steerdist,
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
    icnf::AbstractCondICNF{<:AbstractFloat, <:VectorMode},
    mode::Mode,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
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
            ys,
            ps,
            st;
            resource,
            tspan,
            steerdist,
            basedist,
            differentiation_backend,
            rng,
            sol_args,
            sol_kwargs,
        ),
    )
end

@inline function loss(
    icnf::AbstractCondICNF{<:AbstractFloat, <:MatrixMode},
    mode::Mode,
    xs::AbstractMatrix{<:Real},
    ys::AbstractMatrix{<:Real},
    ps::Any,
    st::Any;
    resource::AbstractResource = icnf.resource,
    tspan::NTuple{2} = icnf.tspan,
    steerdist::Distribution = icnf.steerdist,
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
                ys,
                ps,
                st;
                resource,
                tspan,
                steerdist,
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
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{<:AbstractFloat, <:ADVectorMode},
    mode::TestMode,
    ys::AbstractVector{<:Real},
    ϵ::AbstractVector{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    fnn = first ∘ icnf.nn
    z = @view u[begin:(end - n_aug - 1)]
    mz, J = AbstractDifferentiation.value_and_jacobian(
        differentiation_backend,
        x -> fnn(vcat(x, ys), p, st),
        z,
    )
    trace_J = tr(only(J))
    vcat(mz, -trace_J)
end

function augmented_f(
    u::Any,
    p::Any,
    t::Any,
    icnf::AbstractCondICNF{<:AbstractFloat, <:MatrixMode},
    mode::TestMode,
    ys::AbstractMatrix{<:Real},
    ϵ::AbstractMatrix{<:Real},
    st::Any;
    resource::AbstractResource = icnf.resource,
    differentiation_backend::AbstractDifferentiation.AbstractBackend = icnf.differentiation_backend,
    rng::AbstractRNG = Random.default_rng(),
)
    n_aug = n_augment(icnf, mode)
    fnn = first ∘ icnf.nn
    z = @view u[begin:(end - n_aug - 1), :]
    mz, J = jacobian_batched(icnf, x -> fnn(vcat(x, ys), p, st), z; resource)
    trace_J = transpose(tr.(eachslice(J; dims = 3)))
    vcat(mz, -trace_J)
end

@inline function (icnf::AbstractCondICNF)(xs_ys::Any, ps::Any, st::Any)
    xs, ys = xs_ys
    first(inference(icnf, TrainMode(), xs, ys, ps, st))
end

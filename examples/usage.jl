## Enable Logging
using Logging, TerminalLoggers
global_logger(TerminalLogger())

## Data
using Distributions
ndata = 1024
ndimensions = 1
data_dist = Beta{Float32}(2.0f0, 4.0f0)
r = rand(data_dist, ndimensions, ndata)
r = convert.(Float32, r)

## Parameters
nvariables = size(r, 1)
naugments = nvariables + 1
n_in = nvariables + naugments + 1 # add time concatenation
n_out = nvariables + naugments
n_hidden = n_in * 4

## Model
using ContinuousNormalizingFlows,
    Lux,
    OrdinaryDiffEqAdamsBashforthMoulton,
    FastBroadcast,
    Polyester,
    SciMLLogging,
    SciMLSensitivity,
    ADTypes,
    Zygote,
    # ForwardDiff, # to use JVP
    # LuxCUDA, # To use gpu
    MLDataDevices

icnf = ICNF(;
    nn = Chain(
        Dense(n_in => n_hidden, softplus),
        Dense(n_hidden => n_hidden, softplus),
        Dense(n_hidden => n_out),
    ),
    nvariables = nvariables, # number of variables
    naugments = naugments, # number of augmented dimensions
    nconditions = 0, # number of conditioning inputs
    λ₁ = 1.0f-2, # regulate flow
    λ₂ = 1.0f-2, # regulate volume change
    λ₃ = 1.0f-2, # regulate augmented dimensions
    steer_rate = 1.0f-1, # add random noise to end of the time span
    tspan = (0.0f0, 1.0f0), # time span
    device = cpu_device(), # process data by CPU
    # device = gpu_device(), # process data by GPU
    autonomous = false, # using non-autonomous flow
    inplace = false, # not using the inplace version of functions
    compute_mode = LuxVecJacMatrixMode(AutoZygote()), # process data in batches and use VJP via Zygote
    # compute_mode = LuxJacVecMatrixMode(AutoForwardDiff()), # process data in batches and use JVP via ForwardDiff
    sol_kwargs = (;
        save_everystep = false,
        maxiters = typemax(Int),
        reltol = 1.0f-4,
        abstol = 1.0f-4,
        alg = VCABM(; thread = Threaded()),
        sensealg = InterpolatingAdjoint(;
            checkpointing = true,
            autodiff = true,
            autojacvec = ZygoteVJP(),
        ),
        progress = false,
        verbose = Detailed(),
    ), # pass to the solver
)

## Fit It
using DataFrames, MLJBase, Zygote, ADTypes, OptimizationOptimisers

function opt_callback(state::Any, l::Any)
    if isone(state.iter % 64) # log the loss at each 64 iterations
        println("Iteration: $(state.iter) | Loss: $l")
    end
    return false
end

icnf_mach_fn = "icnf-machine.jls"
if !isfile(icnf_mach_fn)
    df = DataFrame(permutedims(r), :auto)
    model = ICNFModel(;
        icnf,
        optimizers = (
            OptimiserChain(
                WeightDecay(; lambda = 1.0f-4),
                ClipNorm(1.0f0, 2.0f0; throw = true),
                Adam(; eta = 1.0f-3, beta = (9.0f-1, 9.99f-1), epsilon = 1.0f-8),
            ),
        ),
        batchsize = 1024,
        adtype = AutoZygote(),
        sol_kwargs = (;
            epochs = 300,
            callback = opt_callback,
            progress = true,
            verbose = Detailed(),
        ), # pass to the solver
    )
    mach = machine(model, df)
    fit!(mach)
    # CUDA.@allowscalar fit!(mach) # needed for gpu

    MLJBase.save(icnf_mach_fn, mach) # save it
end
mach = machine(icnf_mach_fn) # load it

## Use It
d = ICNFDist(mach, TestMode())
actual_pdf = pdf.(data_dist, vec(r))
estimated_pdf = pdf(d, r)
new_data = rand(d, ndata)

## Evaluate It
using Distances
mad_ = meanad(estimated_pdf, actual_pdf)
msd_ = msd(estimated_pdf, actual_pdf)
tv_dis = totalvariation(estimated_pdf, actual_pdf) / ndata
res_df = DataFrame(; mad_, msd_, tv_dis)
display(res_df)

## Plot It
using CairoMakie
f = Figure()
ax = Axis(f[1, 1]; title = "Result")
lines!(ax, 0.0f0 .. 1.0f0, x -> pdf(data_dist, x); label = "Actual")
lines!(ax, 0.0f0 .. 1.0f0, x -> pdf(d, vcat(x)); label = "Estimated")
axislegend(ax)
save("result-figure.svg", f)
save("result-figure.png", f)

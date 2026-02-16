# Switch To MKL For Faster Computation
using MKL

## Enable Logging
using Logging, TerminalLoggers
global_logger(TerminalLogger())

## Data
using Distributions
ndata = 1024
ndimension = 1
data_dist = Beta{Float32}(2.0f0, 4.0f0)
r = rand(data_dist, ndimension, ndata)
r = convert.(Float32, r)

## Parameters
nvars = size(r, 1)
naugs = nvars + 1
n_in = nvars + naugs

## Model
using ContinuousNormalizingFlows,
    Lux,
    OrdinaryDiffEqAdamsBashforthMoulton,
    Static,
    SciMLSensitivity,
    ADTypes,
    Zygote,
    MLDataDevices

# To use gpu, add related packages
# using LuxCUDA

nn = Chain(Dense(n_in + 1 => n_in, tanh))
icnf = ICNF(;
    nn = nn,
    nvars = nvars, # number of variables
    naugmented = naugs, # number of augmented dimensions
    λ₁ = 1.0f-2, # regulate flow
    λ₂ = 1.0f-2, # regulate volume change
    λ₃ = 1.0f-2, # regulate augmented dimensions
    steer_rate = 1.0f-1, # add random noise to end of the time span
    tspan = (0.0f0, 1.0f0), # time span
    device = cpu_device(), # process data by CPU
    # device = gpu_device(), # process data by GPU
    cond = false, # not conditioning on auxiliary input
    inplace = false, # not using the inplace version of functions
    autonomous = false, # using non-autonomous flow
    compute_mode = LuxVecJacMatrixMode(AutoZygote()), # process data in batches and use Zygote
    sol_kwargs = (;
        save_everystep = false,
        maxiters = typemax(Int),
        reltol = 1.0f-4,
        abstol = 1.0f-8,
        alg = VCABM(; thread = True()),
        sensealg = InterpolatingAdjoint(; checkpointing = true, autodiff = true),
    ), # pass to the solver
)

## Fit It
using DataFrames, MLJBase, Zygote, ADTypes, OptimizationOptimisers

icnf_mach_fn = "icnf_mach.jls"
if ispath(icnf_mach_fn)
    mach = machine(icnf_mach_fn) # load it
else
    df = DataFrame(transpose(r), :auto)
    model = ICNFModel(;
        icnf,
        optimizers = (OptimiserChain(WeightDecay(), Adam()),),
        batchsize = 1024,
        adtype = AutoZygote(),
        sol_kwargs = (; epochs = 300, progress = true), # pass to the solver
    )
    mach = machine(model, df)
    fit!(mach)
    # CUDA.@allowscalar fit!(mach) # needed for gpu

    MLJBase.save(icnf_mach_fn, mach) # save it
end

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
lines!(ax, 0.0f0 .. 1.0f0, x -> pdf(data_dist, x); label = "actual")
lines!(ax, 0.0f0 .. 1.0f0, x -> pdf(d, vcat(x)); label = "estimated")
axislegend(ax)
save("result-fig.svg", f)
save("result-fig.png", f)

# Switch To MKL For Faster Computation
# using MKL

# Enable Logging
using Logging, TerminalLoggers
global_logger(TerminalLogger())

# Data
using Distributions
ndata = 1024
ndimension = 1
data_dist = Beta{Float32}(2.0f0, 4.0f0)
r = rand(data_dist, ndimension, ndata)
r = convert.(Float32, r)

# Parameters
nvars = size(r, 1)
naugs = nvars
# n_in = nvars # without augmentation
n_in = nvars + naugs # with augmentation

# Model
using ContinuousNormalizingFlows,
    Lux, OrdinaryDiffEqDefault, SciMLSensitivity, ADTypes, Zygote, MLDataDevices

# To use gpu, add related packages
# using LuxCUDA, CUDA, cuDNN

nn = Chain(Dense(n_in => 3 * n_in, tanh), Dense(3 * n_in => n_in, tanh))
icnf = construct(
    RNODE,
    nn,
    nvars, # number of variables
    naugs; # number of augmented dimensions
    compute_mode = LuxVecJacMatrixMode(AutoZygote()), # process data in batches and use Zygote
    inplace = false, # not using the inplace version of functions
    device = cpu_device(), # process data by CPU
    # device = gpu_device(), # process data by GPU
    tspan = (0.0f0, 13.0f0), # have bigger time span
    steer_rate = 1.0f-1, # add random noise to end of the time span
    λ₁ = 1.0f-2, # regulate flow
    λ₂ = 1.0f-2, # regulate volume change
    λ₃ = 1.0f-2, # regulate augmented dimensions
    sol_kwargs = (;
        save_everystep = false,
        alg = DefaultODEAlgorithm(),
        sensealg = InterpolatingAdjoint(),
    ), # pass to the solver
)

# Fit It
using DataFrames, MLJBase, Zygote, ADTypes, OptimizationOptimisers
df = DataFrame(transpose(r), :auto)
model = ICNFModel(
    icnf;
    optimizers = (Adam(),),
    n_epochs = 300,
    adtype = AutoZygote(),
    batch_size = 512,
    sol_kwargs = (; progress = true), # pass to the solver
)
mach = machine(model, df)
fit!(mach)
# CUDA.@allowscalar fit!(mach) # needed for gpu
ps, st = fitted_params(mach)

# Store It
using JLD2, UnPack
jldsave("fitted.jld2"; ps, st) # save
@unpack ps, st = load("fitted.jld2") # load

# Use It
d = ICNFDist(icnf, TestMode(), ps, st) # direct way
# d = ICNFDist(mach, TestMode()) # alternative way
actual_pdf = pdf.(data_dist, vec(r))
estimated_pdf = pdf(d, r)
new_data = rand(d, n)

# Evaluate It
using Distances
mad_ = meanad(estimated_pdf, actual_pdf)
msd_ = msd(estimated_pdf, actual_pdf)
tv_dis = totalvariation(estimated_pdf, actual_pdf) / n
res_df = DataFrame(; mad_, msd_, tv_dis)
display(res_df)

# Plot It
using CairoMakie
f = Figure()
ax = Makie.Axis(f[1, 1]; title = "Result")
lines!(ax, 0.0f0 .. 1.0f0, x -> pdf(data_dist, x); label = "actual")
lines!(ax, 0.0f0 .. 1.0f0, x -> pdf(d, vcat(x)); label = "estimated")
axislegend(ax)
save("result-fig.svg", f)
save("result-fig.png", f)

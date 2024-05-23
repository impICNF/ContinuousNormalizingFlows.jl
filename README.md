# ContinuousNormalizingFlows.jl

[![deps](https://juliahub.com/docs/General/ContinuousNormalizingFlows/stable/deps.svg)](https://juliahub.com/ui/Packages/General/ContinuousNormalizingFlows?t=2)
[![version](https://juliahub.com/docs/General/ContinuousNormalizingFlows/stable/version.svg)](https://juliahub.com/ui/Packages/General/ContinuousNormalizingFlows)
[![pkgeval](https://juliahub.com/docs/General/ContinuousNormalizingFlows/stable/pkgeval.svg)](https://juliahub.com/ui/Packages/General/ContinuousNormalizingFlows)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://impICNF.github.io/ContinuousNormalizingFlows.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://impICNF.github.io/ContinuousNormalizingFlows.jl/dev)
[![Build Status](https://github.com/impICNF/ContinuousNormalizingFlows.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/impICNF/ContinuousNormalizingFlows.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/impICNF/ContinuousNormalizingFlows.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/impICNF/ContinuousNormalizingFlows.jl)
[![Coverage](https://coveralls.io/repos/github/impICNF/ContinuousNormalizingFlows.jl/badge.svg?branch=main)](https://coveralls.io/github/impICNF/ContinuousNormalizingFlows.jl?branch=main)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/C/ContinuousNormalizingFlows.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/C/ContinuousNormalizingFlows.html)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

Implementations of Infinitesimal Continuous Normalizing Flows Algorithms in Julia

## Citing

See [`CITATION.bib`](CITATION.bib) for the relevant reference(s).

## Installation

```julia
using Pkg
Pkg.add("ContinuousNormalizingFlows")
```

## Usage

```julia
# Enable Logging
using Logging, TerminalLoggers
global_logger(TerminalLogger())

# Parameters
nvars = 1
naugs = nvars
# n_in = nvars # without augmentation
n_in = nvars + naugs # with augmentation
n = 1024

# Model
using ContinuousNormalizingFlows, Lux, Zygote #, CUDA, ComputationalResources
nn = Chain(Dense(n_in => 3 * n_in, tanh), Dense(3 * n_in => n_in, tanh))
# icnf = construct(RNODE, nn, nvars) # use defaults
icnf = construct(
    RNODE,
    nn,
    nvars, # number of variables
    naugs; # number of augmented dimensions
    compute_mode = DIVecJacMatrixMode(AutoZygote()), # process data in batches
    tspan = (0.0f0, 13.0f0), # have bigger time span
    steer_rate = 1.0f-1, # add random noise to end of the time span
    # resource = CUDALibs(), # process data by GPU
    # inplace = true, # use the inplace version of functions
)

# Data
using Distributions
data_dist = Beta{Float32}(2.0f0, 4.0f0)
r = rand(data_dist, nvars, n)
r = convert.(Float32, r)

# Fit It
using DataFrames, MLJBase #, ForwardDiff, ADTypes, OptimizationOptimisers
df = DataFrame(transpose(r), :auto)
# model = ICNFModel(icnf) # use defaults
model = ICNFModel(
    icnf;
    batch_size = 256, # have bigger batchs
    # n_epochs = 100, # have less epochs
    # optimizers = (Adam(),), # use a different optimizer
    # adtype = AutoForwardDiff(), # use ForwardDiff
)
mach = machine(model, df)
fit!(mach)
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
```

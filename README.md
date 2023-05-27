# ContinuousNormalizingFlows.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://impICNF.github.io/ContinuousNormalizingFlows.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://impICNF.github.io/ContinuousNormalizingFlows.jl/dev)
[![Build Status](https://github.com/impICNF/ContinuousNormalizingFlows.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/impICNF/ContinuousNormalizingFlows.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/impICNF/ContinuousNormalizingFlows.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/impICNF/ContinuousNormalizingFlows.jl)
[![Coverage](https://coveralls.io/repos/github/impICNF/ContinuousNormalizingFlows.jl/badge.svg?branch=main)](https://coveralls.io/github/impICNF/ContinuousNormalizingFlows.jl?branch=main)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

Implementations of Infinitesimal Continuous Normalizing Flows Algorithms in Julia

## Citing

See [`CITATION.bib`](CITATION.bib) for the relevant reference(s).

## Usage

To add this package, we can do it by

```julia
using Pkg
Pkg.add(; url = "https://github.com/impICNF/ContinuousNormalizingFlows.jl")
```

To use this package, here is an example:

```julia
using ContinuousNormalizingFlows
using Distributions, Lux
# using Flux
# using ForwardDiff, Optimization
# using CUDA, ComputationalResources

# Parameters
nvars = 1
n = 1024

# Data
data_dist = Beta(2.0f0, 4.0f0)
r = rand(data_dist, nvars, n)

# Model
nn = Lux.Chain(Lux.Dense(nvars => 4 * nvars, tanh), Lux.Dense(4 * nvars => nvars, tanh)) # use Lux
# nn = Flux.Chain(Flux.Dense(nvars => 4 * nvars, tanh), Flux.Dense(4 * nvars => nvars, tanh)) |> FluxCompatLayer # use Flux

icnf = construct(RNODE, nn, nvars; tspan = (0.0f0, 4.0f0)) # process data one by one
# icnf = construct(RNODE, nn, nvars; tspan = (0.0f0, 4.0f0), compute_mode = ZygoteMatrixMode) # process data in batches
# icnf = construct(RNODE, nn, nvars; tspan = (0.0f0, 4.0f0), array_type = CuArray) # process data by GPU

# Training
using DataFrames, MLJBase
df = DataFrame(transpose(r), :auto)
model = ICNFModel(icnf) # use Zygote
# model = ICNFModel(icnf; adtype = Optimization.AutoForwardDiff()) # use ForwardDiff
# model = ICNFModel(icnf; resource = CUDALibs()) # use GPU
mach = machine(model, df)
fit!(mach)
ps, st = fitted_params(mach)

# Use It
d = ICNFDist(icnf, ps, st)
actual_pdf = pdf.(data_dist, vec(r))
estimated_pdf = pdf(d, r)
new_data = rand(d, n)

# Evaluation
using Distances
mad_ = meanad(estimated_pdf, actual_pdf)
msd_ = msd(estimated_pdf, actual_pdf)
tv_dis = totalvariation(estimated_pdf, actual_pdf) / n
```

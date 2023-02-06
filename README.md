# ICNF.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://impICNF.github.io/ICNF.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://impICNF.github.io/ICNF.jl/dev)
[![Build Status](https://github.com/impICNF/ICNF.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/impICNF/ICNF.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/impICNF/ICNF.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/impICNF/ICNF.jl)
[![Coverage](https://coveralls.io/repos/github/impICNF/ICNF.jl/badge.svg?branch=main)](https://coveralls.io/github/impICNF/ICNF.jl?branch=main)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

Implementations of Infinitesimal Continuous Normalizing Flows Algorithms in Julia

## Citing

See [`CITATION.bib`](CITATION.bib) for the relevant reference(s).

## Usage

To add this package, we can do it by

```julia
using Pkg
Pkg.add(; url = "https://github.com/impICNF/ICNF.jl")
```

To use this package, here is an example:

```julia
using ICNF
using Distributions, Lux
using DifferentialEquations, SciMLSensitivity

# Parameters
nvars = 1
n = 1024

# Data
data_dist = Beta(2.0f0, 4.0f0)
r = rand(data_dist, nvars, n)

# Model
nn = Chain(Dense(nvars => 4 * nvars, tanh), Dense(4 * nvars => nvars, tanh))
icnf = construct(RNODE, nn, nvars; tspan = (0.0f0, 4.0f0))

# Training
using DataFrames, MLJBase
df = DataFrame(transpose(r), :auto)
model = ICNFModel(icnf)
mach = machine(model, df)
fit!(mach)
ps, st = fitted_params(mach)

# Use It
d = ICNFDist(icnf, ps, st)
actual_pdf = pdf.(data_dist, vec(r))
estimated_pdf = pdf(d, r)
new_data = rand(d, n)

# Evaluation
using LinearAlgebra, Distances
n1 = norm(estimated_pdf - actual_pdf, 1) / n
n2 = norm(estimated_pdf - actual_pdf, 2) / n
tv_dis = totalvariation(estimated_pdf, actual_pdf) / n
```

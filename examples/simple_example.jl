using ICNF
using Distributions, Flux
using DifferentialEquations, SciMLSensitivity

# Parameters
nvars = 1
n = 1024

# Data
data_dist = Beta(2.0, 4.0)
r = rand(data_dist, nvars, n)

# Model
nn = Chain(
    Dense(nvars => 4*nvars, tanh),
    Dense(4*nvars => nvars, tanh),
) |> f64
icnf = RNODE{Float64, Array}(nn, nvars; tspan=(0.0, 4.0))

# Training
using DataFrames, MLJBase
df = DataFrame(transpose(r), :auto)
model = ICNFModel(icnf; opt_app=SciMLOptApp())
mach = machine(model, df)
fit!(mach)

# Use It
d = ICNFDist(icnf)
actual_pdf = pdf.(data_dist, vec(r))
estimated_pdf = pdf(d, r)
new_data = rand(d, n)

# Evaluation
using LinearAlgebra, Distances
n1 = norm(estimated_pdf - actual_pdf, 1) / n
n2 = norm(estimated_pdf - actual_pdf, 2) / n
tv_dis = totalvariation(estimated_pdf, actual_pdf) / n

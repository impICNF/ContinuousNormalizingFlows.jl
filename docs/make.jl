using ContinuousNormalizingFlows
using Documenter

DocMeta.setdocmeta!(
    ContinuousNormalizingFlows,
    :DocTestSetup,
    :(using ContinuousNormalizingFlows);
    recursive = true,
)

makedocs(;
    modules = [ContinuousNormalizingFlows],
    authors = "Hossein Pourbozorg <prbzrg@gmail.com> and contributors",
    repo = "https://github.com/impICNF/ContinuousNormalizingFlows.jl/blob/{commit}{path}#{line}",
    sitename = "ContinuousNormalizingFlows.jl",
    format = Documenter.HTML(;
        canonical = "https://impICNF.github.io/ContinuousNormalizingFlows.jl",
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/impICNF/ContinuousNormalizingFlows.jl", devbranch = "main")

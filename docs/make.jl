import Documenter, ContinuousNormalizingFlows

Documenter.DocMeta.setdocmeta!(
    ContinuousNormalizingFlows,
    :DocTestSetup,
    :(using ContinuousNormalizingFlows);
    recursive = true,
)

Documenter.makedocs(;
    modules = [ContinuousNormalizingFlows],
    authors = "Hossein Pourbozorg <prbzrg@gmail.com> and contributors",
    repo = "https://github.com/impICNF/ContinuousNormalizingFlows.jl/blob/{commit}{path}#{line}",
    sitename = "ContinuousNormalizingFlows.jl",
    format = Documenter.HTML(;
        canonical = "https://impICNF.github.io/ContinuousNormalizingFlows.jl",
        edit_link = "main",
    ),
    pages = ["Home" => "index.md"],
)

Documenter.deploydocs(;
    repo = "github.com/impICNF/ContinuousNormalizingFlows.jl",
    devbranch = "main",
)

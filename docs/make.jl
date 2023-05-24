using ContinuousNF
using Documenter

DocMeta.setdocmeta!(ContinuousNF, :DocTestSetup, :(using ContinuousNF); recursive = true)

makedocs(;
    modules = [ContinuousNF],
    authors = "Hossein Pourbozorg <prbzrg@gmail.com> and contributors",
    repo = "https://github.com/impICNF/ContinuousNF.jl/blob/{commit}{path}#{line}",
    sitename = "ContinuousNF.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://impICNF.github.io/ContinuousNF.jl",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/impICNF/ContinuousNF.jl", devbranch = "main")

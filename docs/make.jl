using ICNF
using Documenter

DocMeta.setdocmeta!(ICNF, :DocTestSetup, :(using ICNF); recursive = true)

makedocs(;
    modules = [ICNF],
    authors = "Hossein Pourbozorg <prbzrg@gmail.com> and contributors",
    repo = "https://github.com/impICNF/ICNF.jl/blob/{commit}{path}#{line}",
    sitename = "ICNF.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://impICNF.github.io/ICNF.jl",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/impICNF/ICNF.jl", devbranch = "main")

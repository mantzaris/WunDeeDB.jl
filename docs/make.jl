using WunDeeDB
using Documenter

DocMeta.setdocmeta!(WunDeeDB, :DocTestSetup, :(using WunDeeDB); recursive=true)

makedocs(;
    modules=[WunDeeDB],
    authors="Alexander V. Mantzaris",
    sitename="WunDeeDB.jl",
    format=Documenter.HTML(;
        canonical="https://mantzaris.github.io/WunDeeDB.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mantzaris/WunDeeDB.jl",
)

using Franklin
using Literate
using Tar

process_literate(fstring) = replace(fstring, "_tutorials" => ".")

function build_tutorials(path = "_tutorials", tarpath = "__site/assets/"; ignore = [])
    # create directory to store outputs
    outpath = "__generated"
    isdir(outpath) || mkpath(outpath)

    # copy tomls
    for file in ["Project.toml", "Manifest.toml", "Artifacts.toml"]
        cp(joinpath(path, file), joinpath(outpath, file); force = true)
    end

    # copy src
    isdir(joinpath(outpath, "src")) || mkpath(joinpath(outpath, "src"))
    cp(joinpath(path, "src"), joinpath(outpath, "src"); force = true)

    # copy tutorials
    for file in readdir(path)
        _, ext = splitext(file)
        (ext == ".jl") && (file ∉ ignore) || continue

        Literate.script(joinpath(path, file), outpath;
                        execute = false, preprocess = process_literate)
    end

    # put it all together
    isdir(tarpath) || mkpath(tarpath)
    Tar.create(outpath, joinpath(tarpath, "tutorials.tar.gz"))
end

function preview(; eval_all = false)
    build_tutorials(ignore = ["training.jl"])
    serve(single = true)
    serve(eval_all = eval_all)
end

function deploy()
    build_tutorials(ignore = ["training.jl"])
    optimize()
end

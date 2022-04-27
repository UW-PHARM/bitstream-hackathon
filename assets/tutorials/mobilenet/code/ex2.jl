# This file was generated, do not modify it. # hide
include("_tutorials/src/setup.jl");

artifacts = "_tutorials/Artifacts.toml"
ensure_artifact_installed("mobilenet", artifacts)
mobilenet = artifact_hash("mobilenet", artifacts)
modelpath = joinpath(artifact_path(mobilenet), "mobilenet.bson")
model = BSON.load(modelpath, @__MODULE__)[:m];
# This file was generated, do not modify it. # hide
ensure_artifact_installed("vww", artifacts)
vwwdata = artifact_hash("vww", artifacts)
dataroot = joinpath(artifact_path(vwwdata), "vww-hackathon")
valdata = VisualWakeWords(dataroot; subset = :val)
valaug = map_augmentation(ImageToTensor(), valdata)
valloader = DataLoader(BatchView(valaug; batchsize = 32); buffer = true)

accfn(ŷ::AbstractArray, y::AbstractArray) = mean((ŷ .> 0) .== y)
accfn(data, model) = mean(accfn(model(x), y) for (x, y) in data)

accfn(valloader, model)
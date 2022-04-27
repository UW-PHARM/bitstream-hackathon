# This file was generated, do not modify it. # hide
model_rescaled = Chain(model_scaled, x -> x .* total_scaling)
accfn(valloader, model_rescaled)
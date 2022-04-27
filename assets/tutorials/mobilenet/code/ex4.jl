# This file was generated, do not modify it. # hide
model_scaled, scalings = prepare_bitstream_model(model)
@show total_scaling = prod(prod.(scalings))
model_scaled
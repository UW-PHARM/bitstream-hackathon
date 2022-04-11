relu1(x) = min(relu(x), 1)
relu1_scale(x) = relu1(x / 2)

_rm_bn(layer::Conv, act) =
  Conv(act, layer.weight, layer.bias, layer.stride, layer.pad, layer.dilation, layer.groups)
_rm_bn(layer::BatchNorm, act) = nothing
_rm_bn(layer) = layer

function MobileNet(activation = relu, width_mult = 1; batchnorm = true, kwargs...)
  base = Metalhead.mobilenetv1(width_mult, Metalhead.mobilenetv1_configs;
                               activation = activation, kwargs...)

  if !batchnorm
    backbone_layers = []
    for layer in base[1]
      _layer = _rm_bn(layer)
      !isnothing(_layer) && push!(backbone_layers, _layer)
    end

    return Chain(Chain(backbone_layers...), base[2])
  else
    return base
  end
end

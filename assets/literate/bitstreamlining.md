<!--This file was generated, do not modify it.-->
````julia:ex1
using Pkg # hideall
Pkg.activate("_tutorials/Project.toml")
Pkg.instantiate()
````

# Bitstreamlining: Training a model for bitstream computing

The biggest caveat with bitstream is that all our values need to be restricted within [-1, 1].
We will now go over some of the changes that help a floating point training system
better align with bitstream constraints.

## Activation function

A better fit for our range constraint is to use [hardtanh](https://fluxml.ai/Flux.jl/stable/models/activation/#NNlib.hardtanh)
as the activation function. Apart from being
much cheaper computationally, it would mimic the behavior of not having any explicit activation function
in the bitstream model.

However, you might notice this is not the activation function present in the pretrained model.
Due to the saturating nature of hardtanh, no "learning" might occur once values saturate to a magnitude of 1.
Therefore, we add a slight [negative slope](https://arxiv.org/abs/1603.00391) to hardtanh beyond our range of interest in the function slopehtanh() for training the model.

## Penalizing saturation of parameters

When the floating point model is ported to the bitstream realm, any parameters
larger than 1 in magnitude would saturate their stream. What that means
for our accuracy is, that the parameter has now changed its value and
accuracy might decrease more than anticipated if too many parameters
get saturated during training. One way to tackle this problem is to penalize
saturated parameters during training. We can do this by adding a
[softshrink](https://fluxml.ai/Flux.jl/stable/models/activation/#NNlib.softshrink) (with λ=1) of all
parameters to the loss in training phase, creating a magnitude-aware training scheme.

The functions enable\_shrinkloss and disable\_shrinkloss can help toggle this functionality on and off. It is off by default.

IT IS HIGHLY RECOMMENDED TO HAVE enable_shrinkloss() BEFORE YOU TRAIN YOUR MODEL TO AVOID A DIP IN ACCURACY

## Training without the Batchnorm
While the BatchNorm helps our model train, it can be a place for parameter saturations to hide.
Once we are satisfied with the way our model is trained, we can merge the batchnorms and
remove these saturations, but that can slightly decrease the accuracy. So, we merge the model for a few
epochs to recoup the lost accuracy. Do note that if the dip in accuracy
from validation phase to merging of batchnorms is high, it might be better
to splice in blank batch norm layers and train the model as training
can become more erratic without the batchnorm layers.
On merging batchnorm, it would be helpful to desaturate the model
once so that any saturations just get replaced by -1 or 1s respectively
and do not make the parameter based loss explode.

````julia:ex2
include("_tutorials/src/setup.jl")
BSON.@load "_tutorials/src/pretrained.bson" m
m_merged = merge_conv_bn(m)
m_merged = desaturate(m_merged)
m = rebn(m_merged)
````


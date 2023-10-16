<!--This file was generated, do not modify it.-->
````julia:ex1
using Pkg # hideall
Pkg.activate("_tutorials/Project.toml")
Pkg.instantiate()
````

# Simulating MobileNet

## Evaluating the baseline model

In [Bitstreams 101](/tutorials/bitstream), you saw how we can compute a
multiplication operation using stochastic bitstreams with a single AND gate.
But there are many more operations that we can perform with bitstreams,
including addition, division, vector dot products, and matrix multiplication,
to name a few.
In this tutorial, you will learn how all these operations can come together
to evaluate a neural network using bitstream computing.

First, let us take a pretrained version of our model, MobileNet v1.
In your case, you will prune this model first, then follow this tutorial.

````julia:ex2
include("_tutorials/src/setup.jl");

BSON.@load "_tutorials/src/pretrained.bson" m
````

Let's see the accuracy of the pretrained model on the provided validation data set.

````julia:ex3
ensure_artifact_installed("vww", artifacts)
vwwdata = artifact_hash("vww", artifacts)
dataroot = joinpath(artifact_path(vwwdata), "vww-hackathon")
valdata = VisualWakeWords(dataroot; subset = :val)
valaug = map_augmentation(ImageToTensor(), valdata)
valloader = DataLoader(BatchView(valaug; batchsize = 32); buffer = true)

accfn(ŷ::AbstractArray, y::AbstractArray) = mean((ŷ .> 0) .== y)
accfn(data, model) = mean(accfn(model(x), y) for (x, y) in data)

accfn(valloader, model)
````

This accuracy is quite good at 79%!

## Building the bitstream computing model

At the end of [Bitstreams 101](/tutorials/bitstream), you saw that simulating
bitstream computing circuits at the bit level requires generating a bit for
each input bitstream, then emulating the hardware on those bits, and pushing
the result onto an output bitstream.

BitSAD.jl automates this process with a
[`simulatable` function](https://uw-pharm.github.io/BitSAD.jl/dev/docs/tutorials/simulation-and-hardware.html)
that takes a Julia function and builds a "simulatable" version of it.
Unfortunately, we can't naively apply this function to our model.
For example, our model weights and biases are represented as floating point
numbers > 1. So, we must first prepare our model for bitstream mode.
We have provided you with a utility function that does this step for you.

The function is called `prepare_bitstream_model` and it takes a single argument:
the model. It merges the batch norm and convolution layers into a single convolution layer.
This is done by adjusting the convolution layer weights and biases
according to Eq. 1 below where $w$ and $b$ are the original weights and biases,
$\gamma$ and $\beta$ are the batch norm scale and shift,
and $\mu$ and $\sigma$ are the batch norm running mean and variance.
   $$
   \bar{w} = \frac{w \gamma}{\sigma} \qquad \bar{b} = \frac{\gamma (b - \mu)}{\sigma} + \beta
   $$

## Approximating the simulation error

We see that the total scaling is quite large. This means that we will need
long bitstreams to accurately represent the scaled weights and biases.
Typically, we would use the `simulatable` function in BitSAD to empirically
measure the effect of this error on our accuracy; however, cycle-accurate
simulation of hardware is computationally intensive for a large program like
a neural network. At UW-Madison, our group uses the compute resources available
on campus to simulate these models. For the hackathon, we will be using an
approximation of the error induced by simulating bitstreams.

````julia:ex4
simulation_length = 1000
add_conversion_error!(model_scaled, simulation_length);
````

We provide you with a `add_conversion_error!` function that accepts a model
and simulation length in clock cycles. This function will use BitSAD to measure
the error incurred by generating bit sequences for each weight and bias,
then adjust the floating point weights and biases of the model to be slightly
off by the measured error.
You can then evaluate the adjusted model like you evaluated the original
baseline model above. We make one additional change to the model to make sure
we re-scale the output according to `total_scaling` defined above.
The drop in accuracy will be the penalty paid for the inaccuracy in the bitstreams.

````julia:ex5
model_rescaled = Chain(model_scaled, x -> x .* total_scaling)
accfn(valloader, model_rescaled)
````

The accuracy can vary substantially relative to the original baseline accuracy.
This can happen if we use an extremely short simulation length.
In practice, your simulation length should be on the order of 10,000 cycles or more.
Your goal in the hackthon is to choose a pruning strategy and requested latency
that minimizes energy consumption while maximizing accuracy.

## Real simulation

As we mentioned above, simulating the network using BitSAD will be computationally
intensive, so we do not require this for the hackathon.
Unfortunately, our approximation model does not account for all the possible
sources of error in bitstream computing, namely correlations between the
input bitstreams as well as errors caused by the stateful emulation of the hardware.
BitSAD simulatable functions do account for all these sources of error.
If you are interested in evaluating your model using BitSAD, you can execute
```julia
mbit = model_scaled |> tosbitstream
msim = make_simulatable(mbit, (96, 96, 3, 1))
```
This code does two things. First, the `tosbitstream` function will replace
all the array parameters in your model with `SBitstream` arrays from BitSAD.
Next, the `make_simulatable` function will apply the `simulatable` function
from BitSAD to each layer in the network and wrap the result as a "simulatable"
version of the layer. We pass the `make_simulatable` function the size of our
input, since the hardware will be built of a specific size. Finally, you can
simulate a single cycle of bitstream execution using
```julia
xbit = SBitstream.(x) # convert a single input sample to SBitstream
msim(xbit)
```
The output of `msim(xbit)` will be a single element `SBitstream` array.
If you examine the single element, you will see that it contains a single bit
in the queue. That single bit is the result of simulating the bitstream computing
hardware. You can now call `msim(xbit)` repeatedly in a loop to simulate many
bit sequentially. For more information, check out the
[BitSAD.jl docs](https://uw-pharm.github.io/BitSAD.jl/dev/README.html).


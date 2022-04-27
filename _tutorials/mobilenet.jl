using Pkg # hideall
Pkg.activate("_tutorials/Project.toml")
Pkg.instantiate()

# # Simulating MobileNet

# ## Evaluating the baseline model

# In [Bitstreams 101](/tutorials/bitstream), you saw how we can compute a
# multiplication operation using stochastic bitstreams with a single AND gate.
# But there are many more operations that we can perform with bitstreams,
# including addition, division, vector dot products, and matrix multiplication,
# to name a few.
# In this tutorial, you will learn how all these operations can come together
# to evaluate a neural network using bitstream computing.

# First, let us download a pretrained version of our model, MobileNet v1.
# We will do this using the artifact system shown below.
# In your case, you will prune this model first, then follow this tutorial.

include("_tutorials/src/setup.jl");

artifacts = "_tutorials/Artifacts.toml"
ensure_artifact_installed("mobilenet", artifacts)
mobilenet = artifact_hash("mobilenet", artifacts)
modelpath = joinpath(artifact_path(mobilenet), "mobilenet.bson")
model = BSON.load(modelpath, @__MODULE__)[:m];

# In addition to the training data set and validation data set, we provide
# you with a test data set of only 100 samples. This small subset will be used
# to measure the performance of your model using bitstream computing.
# Let's see the accuracy of the pretrained model on the test data set.

ensure_artifact_installed("vww", artifacts)
vwwdata = artifact_hash("vww", artifacts)
dataroot = joinpath(artifact_path(vwwdata), "vww-hackathon")
valdata = VisualWakeWords(dataroot; subset = :val)
valaug = map_augmentation(ImageToTensor(), valdata)
valloader = DataLoader(BatchView(valaug; batchsize = 32), nothing; buffered = true)

accfn(ŷ::AbstractArray, y::AbstractArray) = mean((ŷ .> 0) .== y)
accfn(data, model) = mean(accfn(model(x), y) for (x, y) in data)

accfn(valloader, model)

# This accuracy is quite good at 81%!

# ## Building the bitstream computing model

# At the end of [Bitstreams 101](/tutorials/bitstream), you saw that simulating
# bitstream computing circuits at the bit level requires generating a bit for
# each input bitstream, then emulating the hardware on those bits, and pushing
# the result onto an output bitstream.
#
# BitSAD.jl automates this process with a
# [`simulatable` function](https://uw-pharm.github.io/BitSAD.jl/dev/docs/tutorials/simulation-and-hardware.html)
# that takes a Julia function and builds a "simulatable" version of it.
# Unfortunately, we can't naively apply this function to our model.
# For example, our model weights and biases are represented as floating point
# numbers > 1. So, we must first prepare our model for bitstream mode.
# We have provided you with a utility function that does this step for you.
#
# The function is called `prepare_bitstream_model` and it takes a single argument:
# the model. It will perform the following steps:
# 1. Merge the batch norm and convolution layers into a single convolution layer.
#    This is done by adjusting the convolution layer weights and biases according to
#    ```math
#    \bar{w} = \frac{w \gamma}{\sigma} \qquad \bar{b} = \frac{\gamma (b - \mu)}{\sigma} + \beta
#    ```
#    where ``w`` and ``b`` are the original weights and biases,
#    ``\gamma`` and ``\beta`` are the batch norm scale and shift,
#    and ``\mu`` and ``\sigma`` are the batch norm running mean and variance.
# 2. Scale the merged weights and biases to be < 1. This is done by finding the
#    largest weight or bias in each layer and normalizing all the parameters by
#    that value (call it ``p_{\text{max}}``).
#    Now, the output of our layer is given by
#    ```math
#    z = \mathrm{relu}\left(\frac{w}{p_{\text{max}}} * x + \frac{b}{p_{\text{max}}}\right)
#    ```
#    In other words, the input to the next layer is scaled down by ``p_{\text{max}}``.
#    We account for this by propagating the scaling factor forward to the next layer,
#    and pre-scale the bias of the next layer by ``p_{\text{max}}``.
#    Then we repeat this process, on the next layer and propagate both scaling
#    factors forward onto the third layer. The final output of the network will
#    be scaled by the product of all the scaling factors at every layer.

model_scaled, scalings = prepare_bitstream_model(model);
total_scaling = prod(prod.(scalings))

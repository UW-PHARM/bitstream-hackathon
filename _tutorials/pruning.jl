using Pkg # hideall
Pkg.activate("_tutorials/Project.toml")
Pkg.instantiate()

# # Pruning tutorial

# ## Introduction

# Deep learning has achieved unprecedented performance on image recognition tasks like ImageNet
# and natural language processing tasks such as question answering and machine translation. These
# models generally are on the order of millions or even billions of parameters. For example, Google's
# recent released large language model, [Pathways Language Model (PaLM)](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html), can do well on tasks like conceptual
# understanding and cause & effect reasoning and contains over 540 Billion parameters!

# While this is great for advancing state of the art (SOTA) in terms of accuracy, we have applications like
# self-driving cars or server-side video processing where we would want to deploy these models 
# on the edge to meet real-time deadlines. By having models on the device, we can avoid the latency cost
# of sending a request to the server for the model to process and sending back the output. 
# In these settings, we can't use these gigantic models for a few reasons: the on-device memory available is 
# quite limited (meaning we can't fit our model into memory) and the number of operations it takes to 
# obtain output from the model would fail constraints like latency and power. Luckily, we can rely on model compression to address these concerns.

# Model compression is the area of research focused on deploying SOTA models in resource-constrained devices while minimizing accuracy
# degradation. Various approaches to compressing a model include: weight pruning, quantization, knowledge distillation, low-rank
# tensor decomposition, hardware-aware neural architecture search, etc. In particular, we will discuss and target, arguably, the simplest 
# of these methods: weight pruning.

# ## Weight Pruning

# ### Unstructured Pruning
# Pruning a deep learning models involves finding a percentage of the weights that don't contribute much to the classification
# output and setting their values to 0. By setting the values to 0, we reduce the memory footprint of the model as well as the 
# number of multiplies and accumulates during inference. The de-facto method is low-weight magnitude pruning where we rank the weights
# in the network and eliminate the smallest weights upto a threshold dictated by a chosen compression ratio. Let's explore 
# how to do this in Julia with the Flux.jl package.

# First, let's begin with some imports that will help us load the dataset and model.
include("_tutorials/src/setup.jl");

# Next, let's define our model. We are using [MobileNetv1](https://arxiv.org/abs/1704.04861), which is a popular deep learning model that achieves
# high classification accuracies while still being very resource-efficient. Note the number of parameters the model
# contains and the amount of memory needed to store the model. We can also calculate the number of multiplies and 
# accumulates that MobileNetv1 incurs to produce an output.

m = MobileNet(relu, 0.25; fcsize = 64, nclasses = 2)
mults, adds, output_size = compute_dot_prods(m, (96, 96, 3, 1)) # height and weight are 96, input channels are 3, batch size = 1
println("MobileNet Mults ", mults, " Adds ", adds)
# 
# Next, we need to load in the dataset to prune and finetune our model.
# show line that loads in the data.
# Now that we've finished our setup, let's prune our model. We can use the FluxPrune.jl package to easily prune the lowest magnitude 
# weights by calling LevelPrune.

m_pruned = prune(LevelPrune(0.1), m);
# FluxPrune's prune function takes in two inputs: the pruning strategy and the model to prune. We are using the LevelPrune
# strategy which traverses each layer of the model and removes the lowest `p%` (`10%` in this case) weights in each layer. This 
# is called unstructured pruning since we are concerned with removing the lowest magnitude weights and not worrying about if
# the sparsity induces some kind of structure. FluxPrune allows you to set a different pruning strategy for every layer in the model
# if you desire. Typically, we also have to finetune our resulting pruned model in order to recover some accuracy penalty induced by 
# setting the weights to 0. Let's compute the number of multiplies and accumulates to see how much we have saved.
#
mults, adds, output_size = compute_dot_prods(m_pruned, (96, 96, 3, 1)) # height and weight are 96, input channels are 3, batch size = 1
println("MobileNet Mults ", mults, " Adds ", adds)
#
# We can see that we have obtained a reduction in the number of multiplies relative to our unpruned baseline. Unstructured
# pruning is powerful in that we are able to prune so aggressively that we can obtain sparse models that perform just as well
# as the baseline at less than `10%` of the original model capacity. While unstructured pruning achieves the best compression vs. accuracy tradeoffs,
# it may not translate into faster inference since the unstructured nature of zeros in the weight matrices may induce irregular memory 
# access patterns and sparse GEMM kernels are competitive with dense ones only at extreme sparsities. For these reasons, one may 
# consider structured pruning instead.

# ### Structured Pruning
# In structured pruning, we remove entire channels (typically) or filters rather than individual weights. This
# type of pruning only applies to the convolutional layers, as the concept of removing structure really applies to conv layers as
# opposed to full-connected layers. By removing the lowest magnitude channels, we are drastically able to reduce the number of multiplies and accumulates
# that our model has to perform. 

# To prune channels, we can define the ChannelPrune strategy, which solely targets the convolutional layers.
m_ch_pruned = prune(ChannelPrune(0.1), m);
mults, adds, output_size = compute_dot_prods(m_ch_pruned, (96, 96, 3, 1)) # height and weight are 96, input channels are 3, batch size = 1
println("MobileNet Mults ", mults, " Adds ", adds)

# Compared to the number of multiplies reduced from unstructured pruning, structured pruning drastically reduces the computational cost incurred by the model during inference.
# The caveat for structured pruning is that by eliminating groups of weights, the compression ratio that structured pruning methods are set at are much lower than those
# from unstructured methods so the memory savings are limited. Choosing what the optimal amount of compression vs. latency of the model during inference is a design choice that must be made during model design and 
# prior to deployment.

# ### Pruning and Finetuning pipeline

# Now that we seen how to prune our model, let's try to finetune it to recover some of the accuracy we lost. First, we need to 
# provide the root directory for our dataset and use it to construct the dataset objects for our training and validation sets. 

#md # ```
#md # dataroot = joinpath(artifact"vww", "vww-hackathon")
#md # traindata = VisualWakeWords(dataroot; subset = :train)
#md # testdata = VisualWakeWords(dataroot; subset = :val)
#md # ```

# Next, we define the data augmentation pipeline to our model. As an aside, you want to include a variety of different augmentations during training as
# it's shown to have increased accuracy and makes the network invariant to those augmentations so it learns the correct features.

#md # ```
#md # augmentations = Rotate(10) |>
#md #                 RandomTranslate((96, 96), (0.05, 0.05)) |>
#md #                 Zoom((0.9, 1.1)) |>
#md #                 ScaleFixed((96, 96)) |>
#md #                 Maybe(FlipX()) |>
#md #                 CenterCrop((96, 96)) |>
#md #                 ImageToTensor()
#md # trainset = map_augmentation(augmentations, traindata)
#md # 
#md # 
#md # testset = map_augmentation(ImageToTensor(), testdata)
#md # ;
#md # ```

# Let's grab the pretrained MobileNet model that we can use to prune, stored in the BSON file.

#md # ```
#md # modelpath = joinpath(artifact"mobilenet", "mobilenet.bson")
#md # m = BSON.load(modelpath)[:m] |> gpu
#md # ```

# We define the dataloader which takes a batch of images from the dataset, which is dictated by our batch size. We defer defining the training dataloader until we have to prune (we'll see why soon).
#md # ```
#md # bs = 32
#md # valloader = DataLoader(BatchView(testset; batchsize = bs), nothing; buffered = true)
#md # ;
#md # ```

# Since the Visual Wake Word dataset is a binary classification problem, we use the binary cross entropy loss, which you can read about in the [Flux.Losses documentation page](https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.logitbinarycrossentropy). We also have to 
# define our accuracy function to determine if we correctly classified the input based on its label.
#md # ```
#md # lossfn = Flux.Losses.logitbinarycrossentropy
#md # accfn(ŷ::AbstractArray, y::AbstractArray) = mean((ŷ .> 0) .== y)
#md # accfn(data, m) = mean(accfn(m(gpu(x)), gpu(y)) for (x, y) in data)
#md # ```

# We are now ready to prune and finetune the model. 

# We use the `iterativeprune` function to progressively prune the model.
# `iterativeprune` accepts 3 arguments: the finetuning function, the pruning strategy to use on the model, and the model to be pruned.
# We define the finetuning function with the [do-block syntax](https://docs.julialang.org/en/v1/manual/functions/#Do-Block-Syntax-for-Function-Arguments) which gets passed to `iterativeprune`
# as an anonymous function.

#md # ```
#md # target_acc = 0.78
#md # nepochs = 5
#md # m̄ = iterativeprune(stages, m) do m̄
#md #     opt = Momentum(0.01)
#md #     ps = Flux.params(m̄)
#md #     subset = random_subset(trainset, 5000) # randomly subsample data to make finetuning faster
#md #     trainloader = DataLoader(BatchView(subset; batchsize = bs), nothing; buffered = true)
#md #     for epoch in 1:nepochs
#md #         @info "Epoch $epoch"
#md #         @time for (x, y) in trainloader
#md #             _x, _y = gpu(x), gpu(y)
#md #             gs = Flux.gradient(ps) do
#md #                 lossfn(m̄(_x), _y)
#md #             end
#md #             Flux.update!(opt, ps, gs)
#md #         end
#md #     end
#md #     GC.gc()
#md #     Flux.CUDA.reclaim()
#md #     @show current_accuracy = accfn(valloader, m̄)
#md #     return current_accuracy > target_acc
#md # end
#md # ```


# We define the optimizer (`SGD` with `Momentum` with a learning rate of `0.01`), training dataloader, and the training loop, as well as the loss and gradient update functions.
# Also, note that we have a function `random_subset` which chooses a smaller random subset of the data to train on. Depending on what computing resources are available, 
# you may find that finetuning on the full dataset can be intensive and potentially a random subset would suffice for our purposes. The exact number to use is a hyperparameter you can play around with.



#md # ```
#md # stages = [
#md #  ChannelPrune(0.1),
#md #  ChannelPrune(0.2),
#md #  ChannelPrune(0.3)    
#md # ]
#md # ```

# `stages` dictates what strategy we should use to the prune the model and by how much. For instance, `stages = [ChannelPrune(0.1), ChannelPrune(0.2)]` means that we are going to apply 2 stages
# of channel pruning in succession until we have a model with `20%` of its channels pruned. `iterativeprune` will apply the pruning to the model and finetune the model for a set number of epochs to reach a target accuracy we predefine. If it doesn't reach the target within the
# specified number of epochs, it will retry for a maximum of five times before giving up and returning the last successful stage.


# In our example, we ultimately want to prune the model by removing `30%` of the channels that have the lowest magnitude and doing this iteratively allows
# the model to recover accuracy more smoothly than if we dropped the channels at once. We can run this code and see that our model is able to reach the target that we had originally set.

 # With this backbone, you should now be able to test out different strategies for pruning the model, potentially at different layers and pruning magnitudes!


# Useful Resources:
# 1. [Blog Post on Pruning and Sparsity](https://intellabs.github.io/distiller/pruning.html)
# 2. [Blog Post on Model Compression](https://medium.com/gsi-technology/an-overview-of-model-compression-techniques-for-deep-learning-in-space-3fd8d4ce84e5)
# 3. [Model Compression Survey Paper](https://arxiv.org/abs/1710.0928)
# 4. [Deep Compression Paper](https://arxiv.org/abs/1510.00149)

Pkg.activate(".") # hideall

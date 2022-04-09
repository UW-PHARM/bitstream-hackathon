using Pkg # hideall
Pkg.activate("_tutorials/Project.toml")
Pkg.instantiate()

## Seperate sections
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
# compute_dot_prods(m (96, 96, 3, 1)) # height and weight are 96, input channels are 3, batch size = 1
# 
# Next, we need to load in the dataset to prune and finetune our model.
# show line that loads in the data.
# Now that we've finished our setup, let's prune our model. We can use the FluxPrune.jl package to easily prune the lowest magnitude 
# weights.

using FluxPrune
m_pruned = prune(LevelPrune(0.1), m)
# FluxPrune's prune function takes in two inputs: the pruning strategy and the model to prune. We are using the LevelPrune
# strategy which traverses each layer of the model and removes the lowest p% (10% in this case) weights in each layer. This 
# is called unstructured pruning since we are concerned with removing the lowest magnitude weights and not worrying about if
# the sparsity induces some kind of structure. FluxPrune allows you to set a different pruning strategy for every layer in the model
# if you desire. Typically, we also have to finetune our resulting pruned model in order to recover some accuracy penalty induced by 
# setting the weights to 0. Let's compute the number of multiplies and accumulates to see how much we have saved.
#
# compute_dot_prods(m_pruned, (96, 96, 3, 1)) # height and weight are 96, input channels are 3, batch size = 1
#
# We can see that we have obtained a reduction in the number of multiplies relative to our unpruned baseline. Unstructured
# pruning is powerful in that we are able to prune so aggressively that we can obtain sparse models that perform just as well
# as the baseline at less than 10% of the original model capacity. While unstructured pruning achieves the best compression vs. accuracy tradeoffs,
# it may not translate into faster inference since the unstructured nature of zeros in the weight matrices may induce irregular memory 
# access patterns and sparse GEMM kernels are competitive with dense ones only at extreme sparsities. For these reasons, one may 
# consider structured pruning instead.

# ### Structured Pruning
# In structured pruning, we remove entire channels (typically) or filters rather than individual weights. This
# type of pruning only applies to the convolutional layers, as the concept of removing structure really applies to conv layers as
# opposed to full-connected layers. By removing the lowest magnitude channels, we are drastically able to reduce the number of multiplies and accumulates
# that our model has to perform. 

# To prune channels, we can define the ChannelPrune strategy, which solely targets the convolutional layers.
m_ch_pruned = prune(ChannelPrune(0.1), m)
# compute_dot_prods(m_pruned, (96, 96, 3, 1)) # height and weight are 96, input channels are 3, batch size = 1

# Compared to the number of multiplies reduced from unstructured pruning, structured pruning drastically reduces the computational cost incurred by the model during inference.
# The caveat for structured pruning is that by eliminating groups of weights, the compression ratio that structured pruning methods are set at are much lower than those
# from unstructured methods so the memory savings are limited. Choosing what the optimal amount of compression vs. latency of the model during inference is a design choice that must be made during model design and 
# prior to deployment.

# Useful Resources:
# 1. [Blog Post on Pruning and Sparsity](https://intellabs.github.io/distiller/pruning.html)
# 2. [Blog Post on Model Compression](https://medium.com/gsi-technology/an-overview-of-model-compression-techniques-for-deep-learning-in-space-3fd8d4ce84e5)
# 3. [Model Compression Survey Paper](https://arxiv.org/abs/1710.0928)
# 4. [Deep Compression Paper](https://arxiv.org/abs/1510.00149)

##

Pkg.activate(".") # hideall
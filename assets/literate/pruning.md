<!--This file was generated, do not modify it.-->
````julia:ex1
using Pkg # hideall
Pkg.activate("_tutorials/Project.toml")
Pkg.instantiate()

# Seperate sections
````

# Pruning tutorial

## Introduction

Deep learning has achieved unprecedented performance on image recognition tasks like ImageNet
and natural language processing tasks such as question answering and machine translation. These
models generally are on the order of millions or even billions of parameters. For example, Google's
recent released large language model, [Pathways Language Model (PaLM)](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html), can do well on tasks like conceptual
understanding and cause & effect reasoning and contains over 540 Billion parameters!

While this is great for advancing state of the art (SOTA) in terms of accuracy, we have applications like
self-driving cars or server-side video processing where we would want to deploy these models
on the edge to meet real-time deadlines. By having models on the device, we can avoid the latency cost
of sending a request to the server for the model to process and the result being back to the device.
In these settings, we can't use these gigantic models for a few reasons: the on-device memory available is
generally a couple hundred megabytes (meaning we can't fit our model into memory) and the number of operations it takes to
obtain output from the model would fail constraints like latency and power. Luckily, we can rely on model compression to address these concerns.

Model compression is the area of research focused on deploying SOTA models in resource-constrained devices while minimizing accuracy
degradation. Various approaches to compressing a model include: weight pruning, quantization, knowledge distillation, low-rank
tensor decomposition, hardware-aware neural architecture search, etc. In particular, we will discuss and target, arguably, the simplest
of these methods: weight pruning.

````julia:ex2
include("_tutorials/src/setup.jl");
````

More stuff

````julia:ex3
m = MobileNet(relu, 0.25; fcsize = 64, nclasses = 2)
#
````

````julia:ex4
Pkg.activate(".") # hideall
````


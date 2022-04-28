@def title = "Submission instructions"

# Submission instructions

Once you have pruned your model, call the `evaluate_submission` function on the floating point model (*note: DO NOT use the bitstream model*). You get to choose the simulation length for evaluating your model.

```julia
include("src/setup.jl")

# after pruning model

# get test set
dataroot = joinpath(artifact"vww", "vww-hackathon")
testdata = VisualWakeWords(dataroot; subset = :test)
testaug = map_augmentation(ImageToTensor(), testdata)
testloader = DataLoader(BatchView(testaug; batchsize = 32), nothing; buffered = true)

# choose simulation length
sim_length = 10_000 # in clock cycles

# evaluate
evaluate_submission(model, testloader, sim_length)
```

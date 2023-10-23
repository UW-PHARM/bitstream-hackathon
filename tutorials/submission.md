@def title = "Submission instructions"

# Submission instructions

Once you have pruned your model, call the `evaluate_submission` function on the floating point model (*note: DO NOT use the bitstream model*). You get to choose the simulation length for evaluating your model.

```julia
include("src/setup.jl")

# choose simulation length
sim_length = 10_000 # in clock cycles

# evaluate your pruned model 
evaluate_submission(model, sim_length)
```

Please use this form to submit your results:
[BCH@UW Submission Form](https://forms.gle/Qqjhmh2F6r3ZBqvB7)

Submissions are due by end of day (any time zone) on Saturday 5/7/2022.

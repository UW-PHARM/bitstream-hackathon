# This file was generated, do not modify it. # hide
es = length(trainloader)
schedule = Interpolator(Step(0.001, 0.5, [20, 10, 20]), es)
optim = Flux.ADAM(0.001)
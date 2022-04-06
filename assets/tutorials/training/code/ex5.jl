# This file was generated, do not modify it. # hide
learner = Learner(m, lossfn;
                  data = (trainloader, valloader),
                  optimizer = optim,
                  callbacks = [schcb, hlogcb, mlogcb, valcb])

# train model

FluxTraining.fit!(learner, 2)
# This file was generated, do not modify it.

using Pkg # hideall
Pkg.activate("_tutorials/Project.toml")
Pkg.update()

# defining the data

dataroot = "/group/ece/ececompeng/lipasti/libraries/datasets/vw_coco2014_96/"
traindata = VisualWakeWords(dataroot; subset = :train) |> shuffleobs
valdata = VisualWakeWords(dataroot; subset = :val)

# data augmentation

augmentations = Rotate(10) |>
                RandomTranslate((96, 96), (0.05, 0.05)) |>
                Zoom((0.9, 1.1)) |>
                ScaleFixed((96, 96)) |>
                Maybe(FlipX()) |>
                CenterCrop((96, 96)) |>
                ImageToTensor()
trainaug = map_augmentation(augmentations, traindata)
valaug = map_augmentation(ImageToTensor(), valdata)
;

# model definition

m = MobileNet(relu, 0.25; fcsize = 64, nclasses = 2)

# data loaders

bs = 32
trainloader = DataLoader(BatchView(trainaug; batchsize = bs), nothing; buffered = true)
valloader = DataLoader(BatchView(valaug; batchsize = 2 * bs), nothing; buffered = true)
;

# training setup

lossfn = Flux.Losses.logitcrossentropy

es = length(trainloader)
schedule = Interpolator(Step(0.001, 0.5, [20, 10, 20]), es)
optim = Flux.ADAM(0.001)

logger = TensorBoardBackend("tblogs")
schcb = Scheduler(LearningRate => schedule)
hlogcb = LogHyperParams(logger)
mlogcb = LogMetrics(logger)
valcb = Metrics(Metric(accuracy; phase = ValidationPhase))

learner = Learner(m, lossfn;
                  data = (trainloader, valloader),
                  optimizer = optim,
                  callbacks = [schcb, hlogcb, mlogcb, valcb])

# train model

FluxTraining.fit!(learner, 2)


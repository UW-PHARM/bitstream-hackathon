# This file was generated, do not modify it. # hide
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
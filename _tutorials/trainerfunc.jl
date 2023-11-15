function trainer(m, func, ln::Real=5)
    dataroot = joinpath(artifact"vww", "vww-hackathon")
    traindata = VisualWakeWords(dataroot; subset = :train) |> shuffleobs
    valdata = VisualWakeWords(dataroot; subset = :val)
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

    bs = 32
    trainloader = DataLoader(BatchView(trainaug; batchsize = bs); buffer = true)
    valloader = DataLoader(BatchView(valaug; batchsize =  2*bs); buffer = true)
    ;

    lossfn = Flux.Losses.logitbinarycrossentropy 
    accfn(ŷ, y) = mean((ŷ .> 0) .== y)
    # define schedule and optimizer
    es = length(trainloader)
    initial_lr = 0.001
    schedule = Interpolator(Step(initial_lr, 0.5, [25, 10]), es)
    optim = func(initial_lr)
    # callbacks

    #logger = TensorBoardBackend("tblog")
    try
        global logger = TensorBoardBackend("tblogs")

        # schcb = Scheduler(LearningRate => schedule)
        global logcb = (LogMetrics(logger),)# LogHyperParams(logger))
        global valcb = Metrics(Metric(accfn; phase = TrainingPhase, name = "train_acc"),
                        Metric(accfn; phase = ValidationPhase, name = "val_acc"))
        global learner = Learner(m, lossfn;
                        data = (trainloader, valloader),
                        optimizer = optim,
                        callbacks = [ToGPU(), logcb..., valcb])

        FluxTraining.fit!(learner, ln)
    finally
        close(logger.logger)
    end

    ## save model
    m = learner.model |> cpu
    return m
end

function trainer(m, ln::Real=10)
    m = trainer(m, Adam, ln)
    return m
end

#m = prune(ChannelPrune(0.1), m)
#m = prune(LevelPrune(0.2), m)
#m = keepprune(m)
#m = prune_propagate(m)
#m = resize_nobn(m)
#m = desaturate(m)
#m = rebn(m)
#m = trainer(m, 1)

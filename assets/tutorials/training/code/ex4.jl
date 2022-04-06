# This file was generated, do not modify it. # hide
logger = TensorBoardBackend("tblogs")
schcb = Scheduler(LearningRate => schedule)
hlogcb = LogHyperParams(logger)
mlogcb = LogMetrics(logger)
valcb = Metrics(Metric(accuracy; phase = ValidationPhase))
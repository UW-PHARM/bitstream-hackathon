# This file was generated, do not modify it. # hide
using Pkg # hideall
Pkg.activate("_tutorials/Project.toml")
Pkg.update()

# defining the data

dataroot = "/group/ece/ececompeng/lipasti/libraries/datasets/vw_coco2014_96/"
traindata = VisualWakeWords(dataroot; subset = :train) |> shuffleobs
valdata = VisualWakeWords(dataroot; subset = :val)

# data augmentation
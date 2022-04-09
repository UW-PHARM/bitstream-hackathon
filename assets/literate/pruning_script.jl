# This file was generated, do not modify it.

using Pkg # hideall
Pkg.activate("_tutorials/Project.toml")
Pkg.instantiate()

# Seperate sections

include("_tutorials/src/setup.jl");

m = MobileNet(relu, 0.25; fcsize = 64, nclasses = 2)

using FluxPrune
m_pruned = prune(LevelPrune(0.1), m)

m_ch_pruned = prune(ChannelPrune(0.1), m)

#

Pkg.activate(".") # hideall


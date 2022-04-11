using Pkg # hideall
Pkg.activate("./Project.toml")
Pkg.instantiate()

include("./src/setup.jl");

m = MobileNet(relu, 0.25; fcsize = 64, nclasses = 2)

using FluxPrune
m_pruned = prune(LevelPrune(0.1), m)

m_ch_pruned = prune(ChannelPrune(0.1), m)

Pkg.activate(".") # hideall

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl


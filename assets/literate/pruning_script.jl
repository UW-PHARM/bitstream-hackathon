# This file was generated, do not modify it.

using Pkg # hideall
Pkg.activate("_tutorials/Project.toml")
Pkg.instantiate()

include("_tutorials/src/setup.jl");

m = MobileNet(relu, 0.25; fcsize = 64, nclasses = 2)


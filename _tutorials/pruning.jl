using Pkg # hideall
Pkg.activate("_tutorials/Project.toml")
Pkg.instantiate()

# # Pruning tutorial

# This is a test.

include("_tutorials/src/setup.jl");

# More stuff

m = MobileNet(relu, 0.25; fcsize = 64, nclasses = 2)

#

Pkg.activate(".") # hideall

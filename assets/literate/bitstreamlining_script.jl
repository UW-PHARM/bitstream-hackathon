# This file was generated, do not modify it.

using Pkg # hideall
Pkg.activate("_tutorials/Project.toml")
Pkg.instantiate()
include("_tutorials/src/setup.jl");

include("_tutorials/src/setup.jl")
BSON.@load "_tutorials/src/pretrained.bson" m
m_merged = merge_conv_bn(m)
m_merged = desaturate(m_merged)
m = rebn(m_merged)
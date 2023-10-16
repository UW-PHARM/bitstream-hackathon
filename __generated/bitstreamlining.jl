using Pkg # hideall
Pkg.activate("./Project.toml")
Pkg.instantiate()
include("./src/setup.jl");

include("./src/setup.jl")
BSON.@load "./src/pretrained.bson" m
m_merged = merge_conv_bn(m)
m_merged = desaturate(m_merged)
m = rebn(m_merged)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

# This file was generated, do not modify it. # hide
include("_tutorials/src/setup.jl")
BSON.@load "_tutorials/src/pretrained.bson" m
m_merged = merge_conv_bn(m)
m_merged = desaturate(m_merged)
m = rebn(m_merged)
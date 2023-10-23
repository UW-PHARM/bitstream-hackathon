# This file was generated, do not modify it. # hide
include("_tutorials/src/setup.jl");
m = MobileNet(slopehtanh, 0.25; fcsize = 64, nclasses = 1);
m_merged = merge_conv_bn(m);
m_merged = desaturate(m_merged);
m_bn = rebn(m_merged);
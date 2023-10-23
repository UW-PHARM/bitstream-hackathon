using Pkg # hideall
Pkg.activate("./Project.toml")
Pkg.instantiate()

include("./src/setup.jl");
m = MobileNet(slopehtanh, 0.25; fcsize = 64, nclasses = 1);
m_merged = merge_conv_bn(m);
m_merged = desaturate(m_merged);
m_bn = rebn(m_merged);

Pkg.activate(".") # hideall

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

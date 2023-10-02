# This file was generated, do not modify it.

using Pkg # hideall
Pkg.activate("_tutorials/Project.toml")
Pkg.instantiate()

include("_tutorials/src/setup.jl");

m = MobileNet(hardtanh, 0.25; fcsize = 64, nclasses = 2)
mults, adds, output_size = compute_dot_prods(m, (96, 96, 3, 1)) # height and weight are 96, input channels are 3, batch size = 1
println("MobileNet Mults ", mults, " Adds ", adds)

m_lv_pruned = prune(LevelPrune(0.1), m);

mults, adds, output_size = compute_dot_prods(m_lv_pruned, (96, 96, 3, 1)) # height and weight are 96, input channels are 3, batch size = 1
println("MobileNet Mults ", mults, " Adds ", adds)

m_ch_pruned = prune(ChannelPrune(0.1), m);
mults, adds, output_size = compute_dot_prods(m_ch_pruned, (96, 96, 3, 1)) # height and weight are 96, input channels are 3, batch size = 1
println("MobileNet Mults ", mults, " Adds ", adds)

m_pruned = keepprune(m_ch_pruned)
m_prop = propagate(m_pruned)
mults, adds, output_size = compute_dot_prods(m_ch_pruned, (96, 96, 3, 1))
println("Propagated MobileNet Mults ", mults, " Adds ", adds)

m_resized = resize(m)

include("trainerfunc.jl");
trainer(m_resized, 2) #trains resized model for 2 epochs

Pkg.activate(".") # hideall


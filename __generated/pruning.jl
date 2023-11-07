using Pkg # hideall
Pkg.activate("./Project.toml")
Pkg.instantiate()

include("./src/setup.jl");

m = MobileNet(slopehtanh, 0.25; fcsize = 64, nclasses = 1)
mults, adds, output_size = compute_dot_prods(m, (96, 96, 3, 1)) # height and weight are 96, input channels are 3, batch size = 1
println("MobileNet Mults ", mults, " Adds ", adds)

m_lv_pruned = prune(LevelPrune(0.2), m);

mults, adds, output_size = compute_dot_prods(m_lv_pruned, (96, 96, 3, 1)) # height and weight are 96, input channels are 3, batch size = 1
println("MobileNet Mults ", mults, " Adds ", adds)

m_ch_pruned = prune(ChannelPrune(0.2), m);
mults, adds, output_size = compute_dot_prods(m_ch_pruned, (96, 96, 3, 1)) # height and weight are 96, input channels are 3, batch size = 1
println("MobileNet Mults ", mults, " Adds ", adds)

m_pruned = keepprune(m_ch_pruned)
m_prop = prune_propagate(m_pruned)
mults, adds, output_size = compute_dot_prods(m_prop, (96, 96, 3, 1))
println("Propagated MobileNet Mults ", mults, " Adds ", adds)

m_resized = resize(m_prop)
mults, adds, output_size = compute_dot_prods(m_resized, (96, 96, 3, 1))
println("Resized MobileNet Mults ", mults, " Adds ", adds)

Pkg.activate(".") # hideall

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
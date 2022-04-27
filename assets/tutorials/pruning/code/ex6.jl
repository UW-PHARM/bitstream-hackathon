# This file was generated, do not modify it. # hide
m_ch_pruned = prune(ChannelPrune(0.1), m);
mults, adds, output_size = compute_dot_prods(m_ch_pruned, (96, 96, 3, 1)) # height and weight are 96, input channels are 3, batch size = 1
println("MobileNet Mults ", mults, " Adds ", adds)
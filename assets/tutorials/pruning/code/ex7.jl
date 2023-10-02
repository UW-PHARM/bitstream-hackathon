# This file was generated, do not modify it. # hide
m_pruned = keepprune(m_ch_pruned)
m_prop = propagate(m_pruned)
mults, adds, output_size = compute_dot_prods(m_ch_pruned, (96, 96, 3, 1))
println("Propagated MobileNet Mults ", mults, " Adds ", adds)
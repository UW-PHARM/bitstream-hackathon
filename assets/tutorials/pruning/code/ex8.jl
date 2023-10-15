# This file was generated, do not modify it. # hide
m_resized = resize(m_prop)
mults, adds, output_size = compute_dot_prods(m_resized, (96, 96, 3, 1))
println("Resized MobileNet Mults ", mults, " Adds ", adds)
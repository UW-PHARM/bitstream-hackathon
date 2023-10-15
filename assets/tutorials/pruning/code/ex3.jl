# This file was generated, do not modify it. # hide
m = MobileNet(slopehtanh, 0.25; fcsize = 64, nclasses = 1)
mults, adds, output_size = compute_dot_prods(m, (96, 96, 3, 1)) # height and weight are 96, input channels are 3, batch size = 1
println("MobileNet Mults ", mults, " Adds ", adds)
# This file was generated, do not modify it. # hide
multiply_sbit(x, y) = SBit((pos(x) * pos(y), neg(x) * neg(y)))

num_samples = 1000
for t in 1:num_samples
    xbit, ybit = pop!(x), pop!(y)
    zbit = multiply_sbit(xbit, ybit)
    push!(z, zbit)
end

abs(estimate(z) - float(z))
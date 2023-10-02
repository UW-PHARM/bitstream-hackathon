
slopehtanh(x::Real) = oftype(x, hardtanh(x) - softshrink(x,1)/4)

function resize(model)
    if(model[1][2] isa BatchNorm)
        return resize_bn(model)
    else
        return resize_nobn(model)

end
#-------------------------------------------------------------------------------------------------

function resize_nobn(model)
    l1new = []
    k=1
    # Loop over each layer in the model[1] (excluding the final layer)
    while k <= length(model[1].layers)
        if(size(l1new,1)==k)
            papalayer = pop!(l1new)
            
        else
            papalayer = deepcopy(model[1].layers[k])
        end

        if(k<length(model[1].layers)-1)
            # Check if the layer is a Conv layer
            if papalayer isa Conv && model[1].layers[k+1] isa Conv && model[1].layers[k+2] isa Conv  

                # if layer is pointwise, next layer depthwise, next to next pointwise again
                if((size((papalayer.weight),1)==1) && (size((model[1].layers[k+2].weight),1)==1) && (size((model[1].layers[k+1].weight),1)==3))
                    t = size(l1new,1)
                    #println("conv PTWISE $k l1 size is $t , and layer+2 and layer+ 4")
                    # Loop over each node in the current layer
                    i=1
                    tempweight1 = Array{Float32}(undef, 1, 1, size(papalayer.weight, 3), 0)
                    tempbias1 = Array{Float32}(undef, 0)
                    tempweight3 = Array{Float32}(undef, 3, 3, 1, 0)
                    tempweight5 = Array{Float32}(undef, 1, 1, 0, size(model[1].layers[k+2].weight, 4))


                    while(i<=size(papalayer.weight, 4))
                        # if this brother is all zero

                        if all(papalayer.weight[:,:, :, i] .== 0)
                            #println("loop removing $i")

                        else 
                            tempweight1 = cat( tempweight1, view(papalayer.weight, :, :, :, i), dims = 4)
                            tempbias1 = cat( tempbias1, view(papalayer.bias, i), dims = 1)
                            tempweight3 = cat( tempweight3, view(model[1].layers[k+1].weight, :, :, :, i), dims = 4)
                            tempweight5 = cat( tempweight5, view(model[1].layers[k+2].weight, :, :, i:i , : ), dims = 3)
                        end
                        i+=1
                    end
                    
                    replacement1 = Conv(tempweight1, tempbias1 ;
                    stride = papalayer.stride,
                    pad = papalayer.pad,
                    dilation = papalayer.dilation,
                    groups = 1)

                    push!(l1new, replacement1)

                    #push!(l1new, BatchNorm(size((tempweight1),4), hardtanh))
                    #k+=2

                    tempbias3 = false
                    s1 = size(tempweight3)
                    s2 = size(tempbias3)
                    #println("tempbias3 being made with $s1, $s2")
                    replacement3 = Conv(tempweight3, tempbias3 ;
                    stride = model[1].layers[k+1].stride,
                    pad = model[1].layers[k+1].pad,
                    dilation = model[1].layers[k+1].dilation,
                    groups = size(tempweight1, 4))

                    push!(l1new, replacement3)
                    #push!(l1new, BatchNorm(size((tempweight1),4), hardtanh))

                    tempbias5 = copy(model[1].layers[k+2].bias)
                    replacement5 = Conv(tempweight5, tempbias5;
                    stride = model[1].layers[k+2].stride,
                    pad = model[1].layers[k+2].pad,
                    dilation = model[1].layers[k+2].dilation,
                    groups = 1)
                    push!(l1new, replacement5)
                    k+=2
                else
                    push!(l1new, papalayer)
                    k+=1
                end
            else
                push!(l1new, papalayer)
                k+=1
            end
        else
            push!(l1new, papalayer)
            k+=1
        end
    end
    return Chain(Chain(l1new...), model[2])
end # resize_nobn

#-------------------------------------------------------------------------------------------------

function resize_bn(model)
    l1new = []
    k=1
    # Loop over each layer in the model[1] (excluding the final layer)
    while k <= length(model[1].layers)
        if(size(l1new,1)==k)
            papalayer = pop!(l1new)
            
        else
            papalayer = deepcopy(model[1].layers[k])
        end

        if(k<length(model[1].layers)-4)
            # Check if the layer is a Conv layer
            if papalayer isa Conv && model[1].layers[k+2] isa Conv && model[1].layers[k+4] isa Conv  

                # if layer is pointwise, next layer depthwise, next to next pointwise again
                if((size((papalayer.weight),1)==1) && (size((model[1].layers[k+4].weight),1)==1) && (size((model[1].layers[k+2].weight),1)==3))
                    t = size(l1new,1)
                    #println("conv PTWISE $k l1 size is $t , and layer+2 and layer+ 4")
                    # Loop over each node in the current layer
                    i=1
                    tempweight1 = Array{Float32}(undef, 1, 1, size(papalayer.weight, 3), 0)
                    tempbias1 = Array{Float32}(undef, 0)
                    tempweight3 = Array{Float32}(undef, 3, 3, 1, 0)
                    tempweight5 = Array{Float32}(undef, 1, 1, 0, size(model[1].layers[k+4].weight, 4))


                    while(i<=size(papalayer.weight, 4))
                        # if this brother is all zero

                        if all(papalayer.weight[:,:, :, i] .== 0)
                            #println("loop removing $i")

                        else 
                            tempweight1 = cat( tempweight1, view(papalayer.weight, :, :, :, i), dims = 4)
                            tempbias1 = cat( tempbias1, view(papalayer.bias, i), dims = 1)
                            tempweight3 = cat( tempweight3, view(model[1].layers[k+2].weight, :, :, :, i), dims = 4)
                            tempweight5 = cat( tempweight5, view(model[1].layers[k+4].weight, :, :, i:i , : ), dims = 3)
                        end
                        i+=1
                    end
                    
                    replacement1 = Conv(tempweight1, tempbias1 ;
                    stride = papalayer.stride,
                    pad = papalayer.pad,
                    dilation = papalayer.dilation,
                    groups = 1)

                    push!(l1new, replacement1)

                    push!(l1new, BatchNorm(size((tempweight1),4), hardtanh))
                    #k+=2

                    tempbias3 = false
                    s1 = size(tempweight3)
                    s2 = size(tempbias3)
                    println("tempbias3 being made with $s1, $s2")
                    replacement3 = Conv(tempweight3, tempbias3 ;
                    stride = model[1].layers[k+2].stride,
                    pad = model[1].layers[k+2].pad,
                    dilation = model[1].layers[k+2].dilation,
                    groups = size(tempweight1, 4))

                    push!(l1new, replacement3)
                    push!(l1new, BatchNorm(size((tempweight1),4), hardtanh))

                    tempbias5 = copy(model[1].layers[k+4].bias)
                    replacement5 = Conv(tempweight5, tempbias5;
                    stride = model[1].layers[k+4].stride,
                    pad = model[1].layers[k+4].pad,
                    dilation = model[1].layers[k+4].dilation,
                    groups = 1)
                    push!(l1new, replacement5)
                    k+=4
                else
                    push!(l1new, papalayer)
                    k+=1
                end
            else
                push!(l1new, papalayer)
                k+=1
            end
        else
            push!(l1new, papalayer)
            k+=1
        end
    end
    return Chain(Chain(l1new...), model[2])
end # resize_bn

#-------------------------------------------------------------------------------------------------

function desaturate(model)
    par, restr = Flux.destructure(model)
    par = hardtanh.(par)
    return restr(par)
end

#-------------------------------------------------------------------------------------------------

function rebn(model)
    out = []
    for layer in model.layers
        if layer isa Chain
            push!(out, rebn(layer))
        elseif layer isa Conv
            newconv = Conv(layer.weight, layer.bias, ; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups)
            newbn = BatchNorm( size(layer.weight,4), layer.σ)
            push!(out,newconv)
            push!(out,newbn)
        else
            push!(out, layer)
        end
    end
    return Chain(out...)
end #rebn

#-------------------------------------------------------------------------------------------------

function fwd0idx(layer)
    #returns a arr of the removable slices for the next layer, made useless by current layer's 0s in conv/dense kernel
    out = []
    if layer isa Conv
        for i in axes(layer.weight, 4)
            # if this brother is all zero
            if all(layer.weight[:,:, :, i] .== 0)
                push!(out,i)
            end
        end
    elseif layer isa Dense
        for i in axes(layer.weight, 1)
            
            if all(layer.weight[i, :] .== 0)
                push!(out,i)
            end
        end
        #println("fwidx ", layer, size(layer.weight, 1), out)
    else
        println("do not try to extract removable elements from non dense / conv layers, byee")
    end
    return out
end

function convfwdidx(in, out, grp, i)
    idx3 = (i-1)%(in÷grp) + 1
    idx4a = (out÷grp)*((i-1)÷(in÷grp)) +1
    idx4b = min((out÷grp)*(((i-1)÷(in÷grp))+1)  , out)
    return idx3, idx4a, idx4b
end

function propfwd(model, rmarr)
    cnt = 0
    for layer in model.layers
        if layer isa Chain
            #recurse for chain elements
            chncnt, rmarr = propfwd(layer, rmarr)
            cnt+=chncnt
        elseif layer isa Dense
            #println("fw layer dense $layer removing $rmarr")
            layer.weight[:, intersect(1:end, rmarr)].=0
            rmarr = fwd0idx(layer)
            cnt+=length(rmarr)
            #println("fw layer dense $layer passing on $rmarr")
        elseif layer isa Conv
            for i in rmarr
                idx3, idx4a, idx4b = convfwdidx( size(layer.weight, 3)*layer.groups, size(layer.weight, 4), layer.groups, i)
                layer.weight[:, :, idx3, idx4a:idx4b].= 0
            end
            rmarr = fwd0idx(layer)
            cnt+=length(rmarr)
        elseif layer isa BatchNorm
            layer.β[intersect(1:end, rmarr)].=0
            layer.γ[intersect(1:end, rmarr)].=0
        end
    end
    #println("cnt is $cnt")
    return cnt, rmarr
end
function propfwd(model)
    out = []
    return propfwd(model, out)
end

#------------------------------------------------------------------------------------------------------------------------------------


function bck0idx(layer)
    #returns a arr of the removable slices for the prev layer, made useless by current layer's 0s in conv/dense kernel
    out = []
    if layer isa Conv
        for i in axes(layer.weight, 3)
            for g in 1:layer.groups
                idx4a, idx4b, newi = convbckidx( size(layer.weight,3), size(layer.weight,4), layer.groups, g, i )
                if all(layer.weight[:, :, i, idx4a:idx4b].== 0)
                    push!(out,newi)
                end
            end
        end
    elseif layer isa Dense
        for i in axes(layer.weight, 2)
            if all(layer.weight[:, i] .== 0)
                push!(out,i)
            end
        end
    else
        println("do not try to extract removable elements from non dense / conv layers, byee")
    end
    return out
end

function convbckidx(thicn, out, grps, g, i)
    idx4a = (out÷grps)*(g-1) +1
    idx4b = (out÷grps)*(g) 
    newi = thicn*(g-1)+i
    #println("thicn $thicn, out $out, grps $grps, g $g i $i idx4a $idx4a idx4b $idx4b, newi $newi")
    return idx4a, idx4b, newi
end

function propbck(model, rmarr)
    cnt = 0
    for layer in reverse(model.layers)
        if layer isa Chain
            #recurse for chain elements
            chncnt, rmarr = propbck(layer, rmarr)
            cnt+=chncnt
        elseif layer isa Dense            
            #println("rev layer dense $layer removing $rmarr")
            layer.weight[intersect(1:end, rmarr), : ].=0
            rmarr = bck0idx(layer)
            cnt+=length(rmarr)
            #println("rev layer dense $layer passing on $rmarr")
        elseif layer isa Conv
            #for i in rmarr
                layer.weight[:, :, :, intersect(1:end, rmarr)].= 0
            #end
            rmarr = bck0idx(layer)
            cnt+=length(rmarr)
        elseif layer isa BatchNorm
            layer.β[intersect(1:end, rmarr)].=0
            layer.γ[intersect(1:end, rmarr)].=0
        end
    end
    #println("cnt is $cnt")
    return cnt, rmarr
end
function propbck(model)
    out = []
    return propbck(model, out)
end

function propagate(model)
    global diff = 1
    global ldiff = 0
    global i = 0
    while(diff!=0)
        global diff = ldiff
        cfwd, _  = propfwd(model)
        cbck, _ = propbck(model)
        global ldiff = cfwd-cbck
        global diff = diff-ldiff
        global i +=1
        #println("i is ", i, " diff is $diff") 
    end
    return model
end
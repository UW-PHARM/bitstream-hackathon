function evaluate_submission(model, data, simulation_length=10000)
    @info "Calculating HW cost..."
    power = power_consumption(model, (96, 96, 3, 1))
    area = area_consumption(model, (96, 96, 3, 1))

    accfn(ŷ::AbstractArray, y::AbstractArray) = mean((ŷ .> 0) .== y)
    accfn(data, model) = mean(accfn(model(x), y) for (x, y) in data)

    @info "Evaluating simulated model performance..."
    if(model[1][2] isa BatchNorm)
        model_scaled = merge_conv_bn(model)
    else
        model_scaled = deepcopy(model)
    end
    #total_scaling = prod(prod.(scalings))
    add_conversion_error!(model_scaled, simulation_length)
    #model_rescaled = Chain(model_scaled, x -> x .* total_scaling)
    acc = accfn(data, model_scaled)

    @info """
    Evaluation complete!

    Area consumption = $area mm²
    Energy consumption = $(power * simulation_length) uW * cycles
    Accuracy = $(round(100 * acc; digits = 2))% correct

    Please submit these results on the website.
    """

    return area, power, acc
end

function evaluate_submission(model, simulation_length=1000)
    dataroot = joinpath(artifact"vww", "vww-hackathon")
    valdata = VisualWakeWords(dataroot; subset = :val)
    valaug = map_augmentation(ImageToTensor(), valdata)
    valloader = DataLoader(BatchView(valaug; batchsize =  2*32), nothing; buffered = true)
    return evaluate_submission(model, valloader, simulation_length)
end

function evaluate_submission_accuracy(model, data, simulation_length)

    accfn(ŷ::AbstractArray, y::AbstractArray) = mean((ŷ .> 0) .== y)
    accfn(data, model) = mean(accfn(model(x), y) for (x, y) in data)

    @info "Evaluating simulated model performance..."
    if(model[1][2] isa BatchNorm)
        model_scaled = merge_conv_bn(model)
    else
        model_scaled = deepcopy(model)
    end
    #total_scaling = prod(prod.(scalings))
    add_conversion_error!(model_scaled, simulation_length)
    #model_rescaled = Chain(model_scaled, x -> x .* total_scaling)
    acc = accfn(data, model_scaled)

    return acc
end

function evaluate_submission_accuracy(model, simulation_length=10000)
    dataroot = joinpath(artifact"vww", "vww-hackathon")
    valdata = VisualWakeWords(dataroot; subset = :val)
    valaug = map_augmentation(ImageToTensor(), valdata)
    valloader = DataLoader(BatchView(valaug; batchsize =  2*32), nothing; buffered = true)
    return evaluate_submission_accuracy(model, valloader, simulation_length)
end
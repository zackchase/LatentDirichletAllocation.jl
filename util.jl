function sample{T <: Number}(numerators::Array{T})
    # Accepts a few numerators of probabilities, and then chooses
    # randomly from the probabilities
    s = sum(numerators)
    value = (rand() * s)
    for i in 1:length(numerators)
        value -= numerators[i]
        if value <= 0.0
            return i
        end
    end
    return 0
end

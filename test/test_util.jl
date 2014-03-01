using Base.Test
using Util

function count(s)
    counts = Int[0, 0, 0, 0]
    for s in samples
        counts[s] += 1
    end
    return counts
end

samples = [sample(Float64[3, 3, 3, 3]) for i in 1:100]
println(count(samples))

samples = [sample(Float64[4, 2, 8, 2]) for i in 1:16000]
println(count(samples))

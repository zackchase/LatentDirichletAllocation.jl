using Base.Test

samples = [sample(Float64[3, 3, 3, 3]) for i in 1:100]
println(count(samples))

samples = [sample(Float64[4, 2, 8, 2]) for i in 1:16000]
println(count(samples))

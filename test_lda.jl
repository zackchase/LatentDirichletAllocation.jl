require("lda.jl")

vocabulary = UTF8String["the", "world", "is", "yours", "globe", "trotter", "123", "234", "789"]

num_documents = 40
documents = speye(length(vocabulary), num_documents)

rng = MersenneTwister(1)

lda = BasicLDA(vocabulary, documents, 3, MersenneTwister(0))
lda_step_random!(lda, rng)

using Base.Test
require("lda.jl")

vocabulary = UTF8String["the", "world", "is", "yours", "globe", "trotter", "123", "234", "789"]

NDOCS = 40
documents = speye(length(vocabulary), NDOCS)

rng = MersenneTwister(1)

lda = BasicLDA(vocabulary, documents, 3, MersenneTwister(0))

@test size(lda.documents, 1) == length(lda.vocabulary)
@test size(lda.topics_, 1) == length(lda.vocabulary)
@test size(lda.assignments_, 1) == length(lda.vocabulary)
@test size(lda.topics_, 2) == size(lda.theta_, 1)
@test size(lda.theta_, 2) == size(lda.assignments_, 2)
@test size(lda.documents, 2) == size(lda.assignments_, 2)

lda_step_random!(lda, rng)

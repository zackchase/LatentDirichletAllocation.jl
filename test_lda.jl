using Base.Test
require("lda.jl")


topic1 = UTF8String["the", 
                    "world",
                    "is",
                    "yours",
                    "globe",
                    "shakespeare"]
topic2 = UTF8String["red",
                    "green",
                    "blue",
                    "yellow",
                    "purple",
                    "orange",
                    "white",
                    "black",
                    "brown"]
topic3 = UTF8String["baseball",
                    "soccer",
                    "basketball",
                    "football",
                    "jersey",
                    "gatorball",
                    "ball"]

vocabulary = vcat(topic1, topic2, topic3)
V = length(vocabulary)
NDOCS = 40

d = Array(Float64, NDOCS, V)
d[1:10,1:7] = 1.0
d[10:23,7:16] = 1.0
d[23:end,16:end] = 1.0
documents = sparse(transpose(d))

rng = MersenneTwister(1)

lda = BasicLDA(vocabulary, documents, 2, rng)

@test size(lda.topics_, 1) == num_topics(lda)
@test size(lda.documents, 2) == num_documents(lda)
@test size(lda.documents, 1) == size_vocabulary(lda)

@test size(lda.documents, 1) == length(lda.vocabulary)
@test size(lda.topics_, 2) == length(lda.vocabulary)
@test size(lda.assignments_, 1) == length(lda.vocabulary)
@test size(lda.topics_, 1) == size(lda.theta_, 1)
@test size(lda.theta_, 2) == size(lda.assignments_, 2)
@test size(lda.documents, 2) == size(lda.assignments_, 2)

maximization_step!(lda)

@printf("Before:\n")
show_topics(STDOUT, lda)
show_documents(STDOUT, lda; documents=2)
lda_step_random!(lda, rng)
@printf("\nAfter:\n")
show_topics(STDOUT, lda)
show_documents(STDOUT, lda; documents=2)

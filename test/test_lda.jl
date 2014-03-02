using Base.Test
using LDA

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

docs = zeros(Float64, NDOCS, V)
docs[1:9,1:6] = 1.0
docs[10:22,7:15] = 1.0
docs[23:40,16:22] = 1.0
#docs = docs[randperm(size(docs, 2)), :]

documents = Array{Int64}[]
for d in 1:NDOCS
    document = Int64[]
    for w in 1:V
        if docs[d, w] > 0
            push!(document, w)
        end
    end
    push!(documents, document)
end


rng = MersenneTwister(1)

lda = BasicLDA(vocabulary, documents, 3, rng)

@test size(lda.topics_, 1) == num_topics(lda)
@test length(lda.documents) == num_documents(lda)

@test size(lda.topics_, 2) == length(lda.vocabulary)
@test size(lda.topics_, 1) == size(lda.theta_, 1)
@test length(lda.documents) == size(lda.theta_, 2)
for d in 1:NDOCS
    @test length(lda.documents[d]) == length(lda.assignments_[d])
end

maximization_step!(lda)

lda.topics_ = sprand(size(lda.topics_, 1), size(lda.topics_, 2), 0.40)

@printf("Before:\n")
show_topics(STDOUT, lda)
println()
show_documents(STDOUT, lda; documents=2)
random_assignment!(lda, rng)
@printf("\nAfter:\n")
show_topics(STDOUT, lda)
println()
show_documents(STDOUT, lda; documents=2)

for i in 1:100
    @printf("\nIteration %i:\n", i)
    gibbs_epoch!(lda, rng)
    show_topics(STDOUT, lda)
    println()
    
    # Maximization step (resets theta and topic distributions)
    maximization_step!(lda)
    show_documents(STDOUT, lda; documents=6)
end


using Base.Test
import LDA: BasicLDA, num_topics, num_documents, 
            show_topics, show_documents, latex_topics,
            random_assignment!, gibbs_epoch!, maximization_step!
using HDF5, JLD

data_dir = ARGS[1]
K = int64(ARGS[2])
@assert ispath(data_dir)

rng = MersenneTwister(42)
vocabulary = readdlm(joinpath(data_dir, "vocabulary.txt"), ' ', UTF8String)[:,1]
docs = @load joinpath(data_dir, "documents.jld")

documents = Array{Int64}[]
for d in 1:400
    document = Int64[]
    for w in 1:length(vocabulary)
        if docs[d, w] > 0
            push!(document, w)
        end
    end
    push!(documents, document)
end

lda = BasicLDA(vocabulary, documents, K, rng)

@test size(lda.topics_, 1) == num_topics(lda)
@test length(lda.documents) == num_documents(lda)

@test size(lda.topics_, 2) == length(lda.vocabulary)
@test size(lda.topics_, 1) == size(lda.theta_, 1)
@test length(lda.documents) == size(lda.theta_, 2)
for d in 1:length(lda.documents)
    @test length(lda.documents[d]) == length(lda.assignments_[d])
end

maximization_step!(lda)

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
    show_documents(STDOUT, lda; documents=2)
end

latex_topics(STDOUT, lda)

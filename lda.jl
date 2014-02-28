abstract LDAStorage

typealias Probability Float64
typealias Vocabulary Array{UTF8String}
typealias Sparse SparseMatrixCSC{Float64,Int64}
typealias Topics Sparse
typealias TopicAssignment Array(Float64, 2)

type BasicLDA <: LDAStorage
    vocabulary::Array{UTF8String}
    documents::Sparse # words x documents

    topics_::Topics # topic probabilities x words
    theta_::Sparse # topic probabilities x documents
    assignments_::Sparse # word x documents
end

function BasicLDA(vocabulary::Array{UTF8String}, documents::Sparse, num_topics::Int, rng::AbstractRNG)
    topics = sprand(num_topics, length(vocabulary), 1.0 / num_topics)
    theta = sprand(num_topics, length(documents), 2.0 / num_topics)
    assignments = ones(length(vocabulary), length(documents))

    BasicLDA(vocabulary, documents, topics, theta, assignments)
end

function num_topics(lda::BasicLDA)
    return length(lda.topics_)
end

function lda_step_random!(lda::BasicLDA, rng::AbstractRNG)
    # Randomly assigns a topic to every word
    for d in 1:length(lda.documents)
        for w in 1:length(lda.documents[d])
            lda.assignments_[w, d] = ceil(rand(rng) * num_topics(lda))
        end
    end
end


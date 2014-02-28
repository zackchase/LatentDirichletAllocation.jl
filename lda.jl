abstract LDAStorage

typealias Probability Float64
typealias Vocabulary Array{UTF8String}
typealias Sparse SparseMatrixCSC{Float64,Int64}

type Topic
    vocabulary::Vocabulary
    probabilities::Array{Probability}
end

type Document
    words::Array{UTF8String}
end

type TopicAssignment
    topics::Array{Topic}
    assignments::Array{Int} # indexes into topics array (all values must be less than length(topic))
end

function TopicAssignment(topics::Array{Topic}, num_words:Int)
    assignments = ones(Int, num_words)
    TopicAssignment(topics, assignments)
end

type BasicLDA <: LDAStorage
    vocabulary::Array{UTF8String}
    documents::Array{Document}

    topics_::Sparse # topic probabilities x words
    theta_::Sparse # topic probabilities x documents
    assignments_::Array{TopicAssignment} # documents-length array of (topic assignments x word)
end

function Topic(vocabulary::Array{UTF8String})
    # Initialize topic to have equal probability for all words in the vocabulary
    probabilities = ones(Probability, length(vocabulary)) / (length(vocabulary))
    Topic(vocabulary, probabilities)
end

function BasicLDA(vocabulary::Array{UTF8String}, documents::Array{Document}, num_topics::Int, rng)
    topics = sprand(num_topics, length(vocabulary), 1.0 / num_topics, rng)
    theta = sprand(num_topics, documents, 2.0 / num_topics, rng)

    assignments = Array{TopicAssignment}
    for i in 1:length(documents)
        a = TopicAssignment(topics, length(documents[i].words))
        push!(assignments, a)
    end

    BasicLDA(vocabulary, documents, topics, theta, assignments)
end

vocabulary = UTF8String["the", "world", "is", "yours", "globe", "trotter", "123", "234", "789"]
documents = Array{Array{UTF8String}}[UTF8String["the", "the", "globe", "trotter"], UTF8String["is", "yours"], UTF8String["123", "234", "789"]
BasicLDA(vocabulary, documents, 3, MersenneTwister(0))

function lda_step_random!(lda::BasicLDA)
    # Randomly assigns a topic to every word
    for d in 1:length(lda.documents)
        for w in 1:length(lda.documents[d].words)
            lda.assignments_[d].assignments[w] = 1
        ends
    end
end


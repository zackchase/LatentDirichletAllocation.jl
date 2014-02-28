abstract LDAStorage

typealias Probability Float64

type Topic
    vocabulary::Array{UTF8String}
    probabilities::Array{Probability}
end

type Document
    words::Array{UTF8String}
end

type TopicAssignment
    topics::Array{Topic}
    assignments::Array{Int} # indexes into topics array (all values must be less than length(topic))
end

type TopicDistribution
    topics::Array{Topic}
    distribution::Array{Probability} # length(topics) == length(distribution)
end

type BasicLDA <: LDAStorage
    vocabulary::Array{UTF8String}
    documents::Array{Document}

    topics_::Array{Topic}
    theta_::Array{TopicDistribution} # topic distributions per document
    assignments_::Array{TopicAssignment} # topic assignments per word
end

function Topic(vocabulary::Array{UTF8String})
    # Initialize topic to have equal probability for all words in the vocabulary
    probabilities = ones(Probability, length(vocabulary)) / (length(vocabulary))
    Topic(vocabulary, probabilities)
end

function TopicDistribution(topics::Array{Topics})
    distribution = Array(Probability, length(topics))
    fill!(distribution, 1.0 / length(topics))
    TopicDistribution(topics, distribution)
end

function TopicAssignment(topics::Array{Topic}, num_words:Int)
    assignments = ones(Int, num_words)
    TopicAssignment(topics, assignments)
end

function BasicLDA(vocabulary::Array{UTF8String}, documents::Array{Document}, num_topics::Int)
    topics = Topic[]
    for i in 1:num_topics
        push!(topics, Topic(vocabulary))
    end

    theta = TopicDistribution[]
    assignments = Array{TopicAssignment}
    for i in 1:length(documents)
        push!(topics, TopicDistribution(topics))

        a = TopicAssignment()
        push(assignments, a)
    end

    BasicLDA(vocabulary, documents, topics, theta, assignments)
end

vocabulary = UTF8String["the", "world", "is", "yours", "globe", "trotter"]
documents = Array{Array{UTF8String}}[UTF8String["the", "the", "globe", "trotter"], UTF8String["is", "yours"]]
BasicLDA(vocabulary, documents, 2)

function random_lda_step(lda::BasicLDA)
    # Randomly assigns a topic to every word
end


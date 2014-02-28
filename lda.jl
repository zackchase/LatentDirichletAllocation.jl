abstract LDAStorage

typealias Probability Float64
typealias Vocabulary Array{UTF8String}
typealias SparseFloats SparseMatrixCSC{Float64,Int64}
typealias Topics SparseFloats
typealias TopicAssignments Matrix{Int64}

type BasicLDA <: LDAStorage
    vocabulary::Array{UTF8String}
    documents::SparseFloats # words x documents

    topics_::Topics # num_topics x num_words
    theta_::SparseFloats # num_topics x num_documents
    assignments_::TopicAssignments # num_words x num_documents
end

function BasicLDA(vocabulary::Array{UTF8String}, documents::SparseFloats, num_topics::Int, rng::AbstractRNG)
    @assert size(documents, 1) == length(vocabulary)
    num_documents = size(documents, 2)
    topics = sprand(num_topics, length(vocabulary), 1.0 / num_topics)
    theta = sprand(num_topics, num_documents, 2.0 / num_topics)
    assignments = ones(length(vocabulary), num_documents)

    BasicLDA(vocabulary, documents, topics, theta, assignments)
end

function num_topics(lda::BasicLDA)
    return size(lda.topics_, 1)
end

function num_documents(lda::BasicLDA)
    return size(lda.documents, 2)
end

function size_vocabulary(lda::BasicLDA)
    return length(lda.vocabulary)
end

function show_topics(lda::BasicLDA; topics=5, words=5)
    for t in 1:min(topics, num_topics(lda))
        top_topics = sort([(t, i) for (i, t) in enumerate(lda.topics_[t,:])], rev=true)
        for w in 1:words
            probability, index = top_topics[w]
            @printf("%i. %s (%f)\n", t, lda.vocabulary[index], probability)
        end
        @printf("\n")
    end
end

function maximization_step!(lda::BasicLDA)
    # Uses assignments to calculate Maximum Likelihood
    #     estimate of topics_
    # Also calculate ML estimate of theta_
    fill!(lda.theta_, 0)
    fill!(lda.topics_, 0)
    for d in 1:num_documents(lda)
        for w in 1:size_vocabulary(lda)
            t = lda.assignments_[w, d]
            lda.topics_[t, w] += 1.0
            lda.theta_[t, d] += 1.0
        end
    end
    
    # TODO: jperla: this can be optimized sparsely
    # now, normalize the counts
    sums = sum(lda.topics_, 2)
    @assert length(sums) == num_topics(lda)
    for t in 1:num_topics(lda)
        s = sums[t]
        if s > 0.0
            for w in 1:size_vocabulary(lda)
                lda.topics_[t, w] /= s
            end
        end
    end    
    
    sums = sum(lda.theta_, 1)
    @assert length(sums) == num_documents(lda)
    for d in 1:num_documents(lda)
        s = sums[d]
        if s > 0.0
            for t in 1:num_topics(lda)
                lda.theta_[t, d] /= s
            end
        end
    end
end

function lda_step_random!(lda::BasicLDA, rng::AbstractRNG)
    # Randomly assigns a topic to every word
    
    # Random Expectation
    for d in 1:num_documents(lda)
        for w in 1:size_vocabulary(lda)
            lda.assignments_[w, d] = ceil(rand(rng) * num_topics(lda))
        end
    end
    
    # Maximization step
    maximization_step!(lda)
end

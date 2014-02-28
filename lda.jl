abstract LDAStorage

typealias Probability Float64
typealias Vocabulary Array{UTF8String}
typealias SparseFloats SparseMatrixCSC{Float64,Int64}
typealias Topics SparseFloats
typealias TopicAssignments Matrix{Int64}

const W = 3
const T = 5

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

function show_topic(io, lda::BasicLDA, t::Int; words=5)
   top_words = sort([(t, i) for (i, t) in enumerate(lda.topics_[t,:])], rev=true)
    @printf(io, "Topic %i:", t) 
    for w in 1:words
      probability, index = top_words[w]
      @printf(io, " %s (%0.2f)", lda.vocabulary[index], probability)
    end
end

function show_topic(lda::BasicLDA, t::Int; words=5)
   io = IOBuffer()
   show_topic(io, lda, t; words=words)
   takebuf_string(io) 
end

function show_topics(io, lda::BasicLDA; topics=5, words=5)
    for t in 1:min(topics, num_topics(lda))
        @printf(io, "%s\n", show_topic(lda, t; words=words))
    end
end

function show_word(io, lda::BasicLDA, w::Int)
    return @printf(io, "%s", lda.vocabulary[w])
end

function show_word(lda::BasicLDA, w::Int)
   io = IOBuffer()
   show_word(io, lda, w)
   takebuf_string(io) 
end

function show_top_words(io, lda::BasicLDA, d::Int; words=5)
    count = 0
    for w in 1:size_vocabulary(lda)
        if lda.documents[w, d] > 0
            @printf(io, " %s", show_word(lda, w))
            count += 1
        end
        if count > words
            break
        end
    end
end

function show_document(io, lda::BasicLDA, d::Int; words=W, topics=T)
    @printf(io, "Document %i:", d)
    show_top_words(io, lda, d; words=words)
    @printf(io, "\n")
    top_topics = sort([(t, i) for (i, t) in enumerate(lda.theta_[:,d])], rev=true)
    for i in 1:min(topics, num_topics(lda))
        proportion, tt = top_topics[i]
        @printf(io, "   %0.2f%%..%s\n", proportion, show_topic(lda, tt; words=words))
    end
end

function show_documents(io, lda::BasicLDA; documents=5, topics=T, words=W)
    for d in 1:min(documents, num_documents(lda))
        show_document(io, lda, d; words=words, topics=topics)
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

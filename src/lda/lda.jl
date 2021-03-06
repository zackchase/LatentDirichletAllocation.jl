include("abstractlda.jl")

import Util: sample

#######################################
# Types
#######################################

typealias TopicIndex Int64
typealias VocabularyIndex Int64
typealias Probability Float64
typealias Word UTF8String

typealias SparseFloats SparseMatrixCSC{Probability, Int64}

typealias Topics SparseFloats
typealias Vocabulary Array{Word}
typealias TopicAssignment Array{TopicIndex}
typealias Document Array{VocabularyIndex}

#######################################
# Constants for keyword arguments
#######################################

const WORDS = 6
const TOPICS = 5
const DOCS = 40

#######################################
# Basic Data Storage
#######################################

type BasicLDA <: LDAStorage
    # Input
    vocabulary::Vocabulary # array of strings
    documents::Array{Document} # words x documents
    alpha::Float64 # parameters
    beta::Float64 # parameters

    # Derived Data during calculations
    topics_::Topics # num_topics x num_words
    theta_::SparseFloats # num_topics x num_documents
    assignments_::Array{TopicAssignment} # num_words x num_documents
end

#######################################
# Constructor for the input
#######################################

function BasicLDA(vocabulary::Array{UTF8String}, documents::Array{Document}, num_topics::Int, rng::AbstractRNG; alpha=0.1, beta=0.1)
    num_documents = length(documents)
    topics = sprand(num_topics, length(vocabulary), 1.0 / num_topics)
    theta = sprand(num_topics, num_documents, 2.0 / num_topics)

    assignments = Array{Int64}[]
    for d in 1:num_documents
        a = ones(Int64, length(documents[d]))
        push!(assignments, a)
    end

    BasicLDA(vocabulary, documents, alpha, beta, topics, theta, assignments)
end

#######################################
# Utility Methods for getting sizes
#######################################

function num_topics(lda::BasicLDA)
    return size(lda.topics_, 1)
end

function num_documents(lda::BasicLDA)
    return length(lda.documents)
end

function length_document(lda::BasicLDA, d::Int64)
    return length(lda.documents[d])
end

function size_vocabulary(lda::BasicLDA)
    return length(lda.vocabulary)
end

#######################################
# Printing to the screen / debugging
#######################################

function show_topic(io, lda::BasicLDA, t::Int; words=WORDS)
    top_words = sort([(t, i) for (i, t) in enumerate(lda.topics_[t,:])], rev=true)
    @printf(io, "Topic %i:", t)
    for w in 1:min(words, size_vocabulary(lda))
      probability, index = top_words[w]
      @printf(io, " %s (%0.4f)", lda.vocabulary[index], probability)
    end
end

function show_topic(lda::BasicLDA, t::Int; words=WORDS)
   io = IOBuffer()
   show_topic(io, lda, t; words=words)
   takebuf_string(io)
end

function show_topics(io, lda::BasicLDA; topics=TOPICS, words=WORDS)
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

function show_top_words(io, lda::BasicLDA, d::Int; words=WORDS)
    for i in 1:min(words, length_document(lda, d))
        @printf(io, " %s", show_word(lda, lda.documents[d][i]))
    end
end

function show_document(io, lda::BasicLDA, d::Int; words=WORDS, topics=TOPICS)
    @printf(io, "Document %i:", d)
    show_top_words(io, lda, d; words=words)
    @printf(io, "\n")
    top_topics = sort([(t, i) for (i, t) in enumerate(lda.theta_[:,d])], rev=true)
    for i in 1:min(topics, num_topics(lda))
        proportion, tt = top_topics[i]
        @printf(io, "   %0.1f%% - %s\n", 100.0 * proportion, show_topic(lda, tt; words=words))
    end
end

function show_documents(io, lda::BasicLDA; documents=DOCS, topics=TOPICS, words=WORDS)
    for d in 1:min(documents, num_documents(lda))
        show_document(io, lda, d; words=words, topics=topics)
    end
end

#######################################
# Latex document table format
#######################################

function latex_topics(io, lda::BasicLDA; words=WORDS)
    #\begin{tabular}{ l c r }
    #  1 & 2 & 3 \\
    #  4 & 5 & 6 \\
    #  7 & 8 & 9 \\
    #\end{tabular}
    T = num_topics(lda)
    max_length = maximum([length(s) for s in lda.vocabulary])
    @printf(io, "\\begin{tabular}{ %s}\n", repeat("c ", T))
    for i in 1:min(words, size_vocabulary(lda))
        @printf(io, " ")
        for t in 1:T
            top_words = sort([(p, i) for (i, p) in enumerate(lda.topics_[t,:])], rev=true)
            @printf(io, " %18s (%0.4f) ", lda.vocabulary[top_words[i][2]], top_words[i][1])
            if t < T
                @printf(io, "&")
            end
        end
        @printf(io, "\\\\\n")
    end
    @printf(io, "\\end{tabular}\n")
end

#######################################
# LDA Implementations
#######################################

#######################################
# M-step in Expectation Maximization
#######################################

function maximization_step!(lda::BasicLDA)
    # Uses assignments to calculate Maximum Likelihood
    #     estimate of topics_
    # Also calculate ML estimate of theta_
    fill!(lda.theta_, 0.0001)
    fill!(lda.topics_, 0.0001)
    for d in 1:num_documents(lda)
        for i in 1:length_document(lda, d)
            t = lda.assignments_[d][i]
            word = lda.documents[d][i]
            lda.topics_[t, word] += 1.0
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

#######################################
# E-Steps (Gibbs, Random, etc)
#######################################

function random_assignment!(lda::BasicLDA, rng::AbstractRNG)
    # Randomly assigns a topic to every word

    # Random Expectation
    for d in 1:num_documents(lda)
        for i in 1:length_document(lda, d)
            lda.assignments_[d][i] = ceil(rand(rng) * num_topics(lda))
        end
    end
end


function sum_words_by_topic(lda::BasicLDA)
    
    count = 0
    K = num_topics(lda)
    sumq = zeros(Int64, K)
    for d in 1:num_documents(lda)
       for i in 1:length_document(lda, d)
          
          current_topic = lda.assignments_[d][i]
          sumq[current_topic] += 1
          count += 1
       end
    end
    
    # println ("The total number of words in all documents is: ", count) 
    # for i =1:K
    #     println ("the number of words for topic $i is: ", sumq[i])
    # end
    
    return sumq
end

function topics_by_words(lda::BasicLDA)

    K = num_topics(lda)
    t_by_w = zeros(Int64, K, size_vocabulary(lda))
    
    for d in 1:num_documents(lda)
        for i in 1:length_document(lda, d)
            current_topic = lda.assignments_[d][i]
            current_word = lda.documents[d][i]
            
            t_by_w[current_topic, current_word] += 1

        end
    end
    return t_by_w
end


function gibbs_epoch!(lda::BasicLDA, rng::AbstractRNG, sumq, tbw)
    # Gibbs assigns a topic to every word
    # TODO: give sample an RNG
    # Random Expectation for all words in the document

    K = num_topics(lda)
    sumbeta = lda.beta * size_vocabulary(lda)
    @assert length(sumq) == K
    
    for d in 1:num_documents(lda)
        for i in 1:length_document(lda, d)
            current_topic = lda.assignments_[d][i]
            
            # println("Current sum is ", sumq[current_topic])
            word = lda.documents[d][i]
            sumq[current_topic] -= 1
            tbw[current_topic, word] -= 1            
            probabilities = zeros(Float64, K)

            for j = 1:K
            
                qjwi = tbw[j, word]

                nprimej = 0
                for w in length(lda.assignments_[d])
                    if lda.assignments_[d][w] == j
                        nprimej +=1
                    end
                end
                prob_1 = (qjwi + lda.beta) / (sumq[j] + sumbeta)
                prob_2 = (nprimej + lda.alpha)
                probabilities[j] = prob_1 * prob_2

            end


            @assert length(probabilities) == K
            assignment = sample(probabilities)
            lda.assignments_[d][i] = assignment
            
            sumq[assignment] += 1
            tbw[assignment, word] += 1            

        end
    end
end

function gibbs_step!(lda::BasicLDA, rng::AbstractRNG)
    # Randomly pick a word to reassign randomly
    d = rand(1:num_documents(lda))
    i = rand(1:length_document(lda, d))

    word = lda.documents[d][i]
    K = num_topics(lda)
    probabilities = (0.01 / K) + full(lda.theta_[:,d] .* lda.topics_[:,word])

    @assert length(probabilities) == K
    assignment = sample(probabilities)
    lda.assignments_[d][i] = assignment
end

function perplexity(lda::BasicLDA)
    sum_prob = 0.0
    sum_words = 0.0
    for d in 1:num_documents(lda)
        for i in 1:length_document(lda, d)
            t = lda.assignments_[d][i]
            word = lda.documents[d][i]
            sum_prob += log(lda.topics_[t, word] * lda.theta_[t, d])
            sum_words += 1.0
        end
    end
    return e^(-1 * sum_prob / sum_words)
end

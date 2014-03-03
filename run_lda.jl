#!/usr/bin/env julia
using Base.Test
import ArgParse: ArgParseSettings, @add_arg_table, parse_args
using HDF5, JLD

import LDA: BasicLDA, num_topics, num_documents, 
            show_topics, show_documents, latex_topics,
            gibbs_epoch!, gibbs_step!,
            random_assignment!, maximization_step!,
            perplexity

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "data_directory"
            help = "where the documents.jld and vocabulary.txt files are"
            arg_type = String
            required = true
        "num_topics"
            help = "the number of topics K to find"
            arg_type = Integer
            required = true
        "--iter", "-i"
            help = "Number of iterations to do"
            arg_type = Integer
            default = 1000
        "--debug_interval", "-x"
            help = "Print debug information every x intervals"
            arg_type = Integer
            default = 100
        "--alpha", "-a"
            help = "alpha parameter to LDA"
            arg_type = FloatingPoint
            default = 0.1
        "--beta", "-b"
            help = "beta parameter to LDA"
            arg_type = FloatingPoint
            default = 0.1
        "--words", "-w"
            help = "number of words to show when debugging"
            arg_type = Integer
            default = 5
        "--docs", "-d"
            help = "number of docs to show when debugging"
            arg_type = Integer
            default = 2 
        "--output", "-o"
            help = "where to output the final data"
            arg_type = String
            default = ""
    end

    return parse_args(s)
end

function best_label(predicted_topics::Array{Int64}, t::Int, true_labels::Array{Int64}, unique_labels::Array{Int64})
    m = 0
    b = 0
    @assert length(predicted_topics) == length(true_labels)
    for label in unique_labels
        if sum([(i == j == true) for (i, j) in zip([t == p for p in predicted_topics], [l == label for l in true_labels])]) > m
            b = label
        end
    end
    return b
end

function accuracy_assigned_labels(lda::BasicLDA, true_labels::Array{Int64})
    # Assign each topic to a label,
    # then return accuracy on true labels.
    theta = lda.theta_
    @assert size(theta, 2) == length(true_labels)
    T = size(theta, 1)
    D = length(true_labels)
    predicted_topics = Int64[]
    for d in 1:D
        push!(predicted_topics, indmax([c for c in theta[:, d]]))
    end
    predicted_labels = zeros(Int64, D)
    unique_labels = unique(true_labels)
    @assert length(predicted_labels) == length(predicted_topics)
    for t in 1:T
        b = best_label(predicted_topics, t, true_labels, unique_labels)
        predicted_labels[[p == t for p in predicted_topics]] = b
    end
    return sum([p == t for (p, t) in zip(predicted_labels, true_labels)]) / D
end

function main()
    parsed_args = parse_commandline()

    data_dir = parsed_args["data_directory"]
    if parsed_args["output"] == ""
        output_dir = joinpath("output", basename(dirname(string(data_dir, "/"))))
    else
        output_dir = parsed_args["output"]
    end
    @assert ispath(data_dir)
    @assert ispath(output_dir)

    K = parsed_args["num_topics"]
    num_iter = parsed_args["iter"]
    words_to_show = parsed_args["words"]
    docs_to_show = parsed_args["docs"]
    debug_interval = parsed_args["debug_interval"]
    alpha, beta = parsed_args["alpha"], parsed_args["beta"]

    rng = MersenneTwister(42)
    vocabulary = readdlm(joinpath(data_dir, "vocabulary.txt"), ' ', UTF8String)[:,1]
    documents_file = joinpath(data_dir, "documents.jld")
    docs = jldopen(documents_file, "r") do file read(file, names(file)[1]) end
    
    @assert size(docs, 2) == length(vocabulary)
    
    documents = Array{Int64}[]
    for d in 1:size(docs, 1)
        document = Int64[]
        for w in 1:length(vocabulary)
            if docs[d, w] > 0
                push!(document, w)
            end
        end
        push!(documents, document)
    end

    true_labels_filename = joinpath(data_dir, "truelabels.txt")
    if ispath(true_labels_filename)
        true_labels = Int64[a for a in readdlm(true_labels_filename, ' ')]
    else
        true_labels = ones(Int64, length(documents))
    end
    
    lda = BasicLDA(vocabulary, documents, K, rng; alpha=alpha, beta=beta)
    
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
    show_documents(STDOUT, lda; documents=docs_to_show)
    random_assignment!(lda, rng)
    @printf("\nAfter:\n")
    show_topics(STDOUT, lda)
    println()
    show_documents(STDOUT, lda; documents=docs_to_show)
    
    perplexities = Float64[]
    accuracies = Float64[]
    for i in 1:num_iter
        gibbs_epoch!(lda, rng)
        p = perplexity(lda)
        a = accuracy_assigned_labels(lda, true_labels)
        push!(perplexities, p)
        push!(accuracies, a)
        if i % debug_interval == 1
            @printf("\nIteration %i (accuracy: %f, perplexity: %f):\n", i, a, p)
            show_topics(STDOUT, lda; words=words_to_show)
            println()
            # Maximization step (resets theta and topic distributions)
            maximization_step!(lda)
            show_documents(STDOUT, lda; documents=docs_to_show, words=words_to_show)
        end
    end

    latex_topics(STDOUT, lda; words=words_to_show)

    # Save final theta if appropriate
    writedlm(joinpath(output_dir, "theta"), full(lda.theta_), ',')
    writedlm(joinpath(output_dir, "topics"), full(lda.topics_), ',')
    writedlm(joinpath(output_dir, "assignments"), full(lda.assignments_), ',')
    writedlm(joinpath(output_dir, "perplexities"), perplexities, ',')
    writedlm(joinpath(output_dir, "accuracies"), accuracies, ',')
    writedlm(joinpath(output_dir, "args"), [ARGS], ',')
end

main()

#!/usr/bin/env julia
using Base.Test
import ArgParse: ArgParseSettings, @add_arg_table, parse_args
using HDF5, JLD

import LDA: BasicLDA, num_topics, num_documents, 
            show_topics, show_documents, latex_topics,
            gibbs_epoch!, gibbs_step!,
            random_assignment!, maximization_step!

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
        "--theta"
            help = "filename of where to save theta at the end of the run"
            arg_type = String
            default = ""
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    data_dir = parsed_args["data_directory"]
    K = parsed_args["num_topics"]
    num_iter = parsed_args["iter"]
    words_to_show = parsed_args["words"]
    docs_to_show = parsed_args["docs"]
    alpha, beta = parsed_args["alpha"], parsed_args["beta"]
    theta_filename = parsed_args["theta"]
    @assert ispath(data_dir)
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
    
    for i in 1:num_iter
        gibbs_step!(lda, rng)
        if i % 100 == 1
            @printf("\nIteration %i:\n", i)
            show_topics(STDOUT, lda; words=words_to_show)
            println()
            # Maximization step (resets theta and topic distributions)
            maximization_step!(lda)
            show_documents(STDOUT, lda; documents=docs_to_show, words=words_to_show)
        end
    end

    latex_topics(STDOUT, lda; words=words_to_show)

    # Save final theta if appropriate
    if theta_filename != ""
        writedlm(theta_filename, full(lda.theta_), ',')
    end
end

main()

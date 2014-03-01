include("lda/abstractlda.jl")                                                                                                                                         
                                                                                                                                                                            
module LDA                                                                                                                                                         
                                                                                                                                                                            
importall Base                                                                                                                                                              
                                                                                                                                                                            
export BasicLDA,                                                                                                                                                     
       num_topics, num_documents, length_document, size_vocabulary,
       show_topic, show_topics, show_word, show_top_words, show_document, show_documents,
       maximization_step!, random_assignment!, gibbs_epoch!
 
include("lda/lda.jl")
                                                                                                                                                                            
end # module LDA

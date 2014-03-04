LatentDirichletAllocation.jl
============================

Implementation of Latent Dirichlet Allocation in Julia. 

> ./run_lda.jl data/simple_test/ 3 -d 0 -i 200 -x 2
> julia plot.jl output/simple_test/theta data/simple_test/truelabels.txt 

> ./run_lda.jl data/classic/ 3 -d 0 -i 200 -x 2
> julia plot.jl output/classic/theta data/classic/truelabels.txt 

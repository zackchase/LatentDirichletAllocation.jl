import PyCall: @pyimport

@pyimport scipy.io as io

# ("data/classic400.mat")
function load_lda_data(filename)
  vars = io.loadmat(filename)

  words = UTF8String[]
  for i in 1:length(vars["classicwordlist"])
    word = convert(Array{UTF8String, 1}, vars["classicwordlist"][i])[1]
    push!(words, word)
  end

  return words
end

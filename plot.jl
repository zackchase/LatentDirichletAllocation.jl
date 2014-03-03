using PyPlot

name = ARGS[1]
true_labels = readdlm(ARGS[2], ' ')
data = readdlm(string(name), ',')

unique_labels = zip(unique(true_labels), ["b", "r", "g", "y", "o"])
unique_map = Dict{Any, String}()
for (u,c) in unique_labels
    unique_map[u] = c
end
colors = String[]
for label in true_labels
    push!(colors, unique_map[label])
end

scatter3D(data[1,:], data[2,:], data[3,:],
          c=colors
)

savefig(string(name, ".png"))

using PyPlot

theta_filename = ARGS[1]
true_labels_filename = ARGS[2]

data = readdlm(theta_filename, ',')
true_labels = readdlm(true_labels_filename, ' ')

unique_labels = zip(unique(true_labels), ["b", "r", "g", "y", "o"])
unique_map = Dict{Any, String}()
for (u,c) in unique_labels
    unique_map[u] = c
end
colors = String[]
for label in true_labels
    push!(colors, unique_map[label])
end

N = length(data[3,:])

scatter3D(data[1,:], data[2,:], data[3,:][1], # + Float64[(i/N) for i in 1:N], # noise to separate, probably not needed
          c=colors
)

savefig(string(theta_filename, ".png"))

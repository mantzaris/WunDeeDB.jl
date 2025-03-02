
"""
    supported_distance_metrics()

Returns a list of the currently supported distance metrics.
"""
function supported_distance_metrics()
    return ["euclidean", "sqeuclidean", "cityblock", "cosine", "jaccard", "chebyshev"]
end

"""
    compute_distance(vec1::AbstractVector, vec2::AbstractVector, metric::String)

Computes the distance between two vectors using the specified metric from Distances.jl.

# Arguments
- `vec1`: First input vector
- `vec2`: Second input vector
- `metric`: String specifying the distance metric to use

# Returns
- The computed distance as a Float64

# Throws
- `ErrorException` if the metric is not supported

# Example
```julia
using Distances  # ensure you have Distances.jl installed

x = [1.0, 2.0, 3.0]
y = [4.0, 5.0, 6.0]

d1 = compute_distance(x, y, "euclidean")   # Euclidean distance
d2 = compute_distance(x, y, "sqeuclidean")   # Squared Euclidean distance
d3 = compute_distance(x, y, "cosine")        # Cosine distance

println("Euclidean: ", d1)
println("Squared Euclidean: ", d2)
println("Cosine: ", d3)
```
"""
function compute_distance(vec1::AbstractVector, vec2::AbstractVector, metric::String)
    length(vec1) == length(vec2) || throw("Vectors must have the same length")

    metric = lowercase(metric)

    if metric == "euclidean"
        return euclidean(vec1, vec2) 
    elseif metric == "sqeuclidean"
        return sqeuclidean(vec1, vec2)
    elseif metric == "cityblock"
        return cityblock(vec1, vec2)
    elseif metric == "cosine"
        return cosine_dist(vec1, vec2)
    elseif metric == "jaccard"
        return jaccard(vec1, vec2)
    elseif metric == "chebyshev"
        return chebyshev(vec1, vec2)
    else
        error("Unsupported metric: $metric. Use one of $(supported_distance_metrics())")
    end
end

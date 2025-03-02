
"""
    supported_distance_metrics()

Returns a list of the currently supported distance metrics.
"""
function supported_distance_metrics()
    return ["euclidean", "sqeuclidean", "cityblock", "cosine", "jaccard", "chebyshev"]
end


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

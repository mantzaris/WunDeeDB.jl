


"""
    supported_distance_metrics()

Returns a list of the currently supported distance metrics, such as ["euclidean", "cosine"].
"""
supported_distance_metrics() = ["euclidean", "cosine"]

#Euclidean distance
function euclidean_distance(vec1::AbstractVector, vec2::AbstractVector)
    return sqrt(sum((vec1 .- vec2).^2))
end

#Cosine similarity
function cosine_similarity(vec1::AbstractVector, vec2::AbstractVector)
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))
end

function compute_distance(vec1::AbstractVector, vec2::AbstractVector, metric::String)
    if metric == "euclidean"
        return euclidean_distance(vec1, vec2)
    elseif metric == "cosine"
        #cosine_similarity is similarity (higher = closer) make it a "distance" by subtracting from 1
        return 1 - cosine_similarity(vec1, vec2)
    else
        error("Unsupported metric: $metric. Use one of $(supported_distance_metrics())")
    end
end

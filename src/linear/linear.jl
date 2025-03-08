

"""
    linear_search_iteration(db_path::String, query_embedding::AbstractVector, metric::String; top_k::Int=5)

Performs a brute-force linear search by iterating over each embedding in the database using
`get_adjacent_id`. Computes the distance to `query_embedding` according to `metric` and
returns the top `top_k` nearest results, sorted by ascending distance. Each result is a 
tuple `(distance, id_text)`.
"""
function linear_search_iteration(db_path::String, query_embedding::AbstractVector, metric::String; top_k::Int=5)
    pq = BinaryMaxHeap{Tuple{Float64,String}}()

    current_id = get_minimum_id(db_path)
    while current_id !== nothing
        emb_dict = get_embeddings(db_path, current_id)

        if haskey(emb_dict,current_id) == false
            return nothing
        end
        
        emb = emb_dict[current_id]

        dist = compute_distance(query_embedding, emb, metric)
        if length(pq) < top_k
            push!(pq, (dist, current_id))
        elseif dist < DataStructures.first(pq)[1]
            pop!(pq)
            push!(pq, (dist, current_id))
        end

        current_id = get_adjacent_id(db_path, current_id, direction="next", full_row=false)
    end

    results = extract_all!(pq)
    return sort(results, by = x -> x[1])
end

"""
    linear_search_ids(db_path::String, query_embedding::AbstractVector, metric::String; top_k::Int=5)

Performs a brute-force linear search by fetching all IDs at once (`get_all_ids`) and 
computing the distance to `query_embedding` for each. Maintains the `top_k` closest 
results, returning them sorted by ascending distance as tuples `(distance, id_text)`.

# Example

```julia
results = linear_search_ids("my_database.sqlite", my_query_embedding, "euclidean"; top_k=5)
for (dist, id) in results
    println("ID: ", id, "  Distance: ", dist)
end
```

"""
function linear_search_ids(db_path::String, query_embedding::AbstractVector, metric::String; top_k::Int=5)
    all_ids = get_all_ids(db_path)
    pq = BinaryMaxHeap{Tuple{Float64,String}}()
    
    for current_id in all_ids
        emb_dict = get_embeddings(db_path, current_id)

        
        if haskey(emb_dict,current_id) == true
            emb = emb_dict[current_id]
            dist = compute_distance(query_embedding, emb, metric)

            if length(pq) < top_k
                push!(pq, (dist, current_id))
            elseif dist < DataStructures.first(pq)[1]
                pop!(pq)
                push!(pq, (dist, current_id))
            end
        end
    end

    results = extract_all!(pq)
    return sort(results, by = x -> x[1])
end



"""
    linear_search_ids_batched(db_path::String, query_embedding::AbstractVector, metric::String; top_k::Int=5, batch_size::Int=1000)

Performs a batched brute-force linear search over embeddings stored in the database at `db_path`.  
The function retrieves all IDs using `get_all_ids` and then processes the embeddings in batches  
(of size `batch_size`). For each batch, it computes the distance from each embedding to the  
`query_embedding` using the specified `metric` (e.g., "euclidean" or "cosine"). It maintains  
the top `top_k` nearest results using a max-heap and returns a vector of tuples `(distance, id_text)`  
sorted in ascending order by distance.

If an error occurs during the retrieval of embeddings (for example, if `get_embeddings` returns an  
error message), the function immediately returns that error message as a String.

# Arguments
- `db_path::String`: Path to the SQLite database file.
- `query_embedding::AbstractVector`: The query embedding vector.
- `metric::String`: The distance metric to use (e.g., "euclidean", "cosine").
- `top_k::Int=5`: (Optional) The number of nearest neighbors to return.
- `batch_size::Int=1000`: (Optional) The number of IDs to process in each batch.

# Returns
- A vector of `(distance, id_text)` tuples sorted by ascending distance, or a String containing  
  an error message if retrieval fails.

# Example
```julia
query = Float32[0.5, 0.5, 0.5]
results = linear_search_ids_batched("my_database.sqlite", query, "euclidean"; top_k=3, batch_size=100)
for (dist, id) in results
    println("ID: ", id, "  Distance: ", dist)
end
```
"""
function linear_search_ids_batched(db_path::String, query_embedding::AbstractVector, metric::String;
    top_k::Int=5, batch_size::Int=1000)

    all_ids = get_all_ids(db_path)
    n = length(all_ids)
    pq = BinaryMaxHeap{Tuple{Float64, String}}()

    i = 1
    while i <= n
        end_i = min(i + batch_size - 1, n)
        ids_chunk = all_ids[i:end_i]

        chunk_embs = get_embeddings(db_path, ids_chunk)

        # If chunk_embs is a String, assume it's an error message.
        if chunk_embs isa String
            return "Error retrieving embeddings: $chunk_embs"
        end

        # If a single ID was requested, ensure we wrap its result into a dictionary.
        if !(chunk_embs isa AbstractDict) && length(ids_chunk) == 1
            chunk_embs = Dict(ids_chunk[1] => chunk_embs)
        end

        # Process each (id, embedding) pair.
        for (the_id, emb) in chunk_embs
            dist = compute_distance(query_embedding, emb, metric)
            if length(pq) < top_k
                push!(pq, (dist, the_id))
            elseif dist < DataStructures.first(pq)[1]
                pop!(pq)
                push!(pq, (dist, the_id))
            end
        end

        i = end_i + 1
    end

    results = extract_all!(pq)
    return sort(results, by = x -> x[1])
end


"""
    linear_search_all_embeddings(db_path::String, query_embedding::AbstractVector, metric::String; top_k::Int=5)

Performs a brute-force linear search over all embeddings stored in the database located at `db_path`.  
For each embedding, the function computes the distance to the provided `query_embedding` using the specified  
`metric` (e.g., "euclidean" or "cosine"). It then maintains the top `top_k` nearest results using a max-heap  
and returns a vector of tuples `(distance, id_text)`, sorted in ascending order by distance.

If `get_all_embeddings(db_path)` returns an error message (as a String), this function immediately returns that error.

# Arguments
- `db_path::String`: Path to the SQLite database file.
- `query_embedding::AbstractVector`: The query embedding vector.
- `metric::String`: The distance metric to use (e.g., "euclidean", "cosine").
- `top_k::Int=5`: (Optional) The number of nearest neighbors to return.

# Returns
- A vector of `(distance, id_text)` tuples sorted by ascending distance, or an error message String if retrieval fails.

# Example
```julia
query = Float32[0.5, 0.5, 0.5]
results = linear_search_all_embeddings("my_database.sqlite", query, "euclidean"; top_k=3)
for (dist, id) in results
    println("ID: ", id, "  Distance: ", dist)
end
```
"""
function linear_search_all_embeddings(db_path::String, query_embedding::AbstractVector, metric::String; top_k::Int=5)
    all_embs = get_all_embeddings(db_path)
    
    # If get_all_embeddings returned an error message, return it immediately.
    if all_embs isa String
        return all_embs
    end

    pq = BinaryMaxHeap{Tuple{Float64, String}}()
    for (id, emb) in all_embs
        dist = compute_distance(query_embedding, emb, metric)
        if length(pq) < top_k
            push!(pq, (dist, id))
        elseif dist < DataStructures.first(pq)[1]
            pop!(pq)
            push!(pq, (dist, id))
        end
    end
    
    results = extract_all!(pq)
    return sort(results, by = x -> x[1])
end



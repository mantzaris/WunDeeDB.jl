


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
        emb = get_embeddings(db_path, current_id)
        if emb !== nothing
            dist = compute_distance(query_embedding, emb, metric)
            if length(pq) < top_k
                push!(pq, (dist, current_id))
            elseif dist < peek(pq)[1]
                pop!(pq)
                push!(pq, (dist, current_id))
            end
        end
        current_id = get_adjacent_id(db_path, current_id, direction="next", full_row=false)
    end

    return sort(collect(pq), by=x->x[1])
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

"""function linear_search_ids(db_path::String, query_embedding::AbstractVector, metric::String; top_k::Int=5)
    all_ids = get_all_ids(db_path)
    pq = BinaryMaxHeap{Tuple{Float64,String}}()
    
    for current_id in all_ids
        emb = get_embeddings(db_path, current_id)
        if emb !== nothing
            dist = compute_distance(query_embedding, emb, metric)

            if length(pq) < top_k
                push!(pq, (dist, current_id))
            elseif dist < peek(pq)[1]
                pop!(pq)
                push!(pq, (dist, current_id))
            end
        end
    end
    
    return sort(collect(pq), by=x->x[1])
end

"""
    linear_search_ids_batched(db_path::String, query_embedding::AbstractVector, metric::String;
                              top_k::Int=5, batch_size::Int=1000)

Performs a brute-force linear search in batches. All IDs are fetched, then processed 
in chunks of size `batch_size` to reduce the overhead of multiple small queries. 
Returns the `top_k` nearest neighbors as `(distance, id_text)` tuples, sorted by 
ascending distance.

# Example

```julia
results = linear_search_ids_batched("my_database.sqlite", my_query_embedding, "cosine"; top_k=5, batch_size=500)
for (dist, id) in results
    println("ID: ", id, "  Distance: ", dist)
end
```

"""function linear_search_ids_batched(db_path::String, query_embedding::AbstractVector, metric::String;
    top_k::Int=5, batch_size::Int=1000)

    all_ids = get_all_ids(db_path)
    n = length(all_ids)

    pq = BinaryMaxHeap{Tuple{Float64,String}}()

    i = 1
    while i <= n
        end_i = min(i + batch_size - 1, n)
        ids_chunk = all_ids[i:end_i]

        chunk_embs = get_embeddings(db_path, ids_chunk)

        if chunk_embs isa String  # an error string
            return "Error retrieving embeddings: $chunk_embs"
        end

        for (the_id, emb) in chunk_embs
            dist = compute_distance(query_embedding, emb, metric)
            if length(pq) < top_k
                push!(pq, (dist, the_id))
            elseif dist < peek(pq)[1]
                pop!(pq)
                push!(pq, (dist, the_id))
            end
        end

        i = end_i + 1
    end

    return sort(collect(pq), by = x -> x[1])
end


"""
    linear_search_all_embeddings(db_path::String, query_embedding::AbstractVector, metric::String; top_k::Int=5)

Performs a brute-force search by loading all embeddings at once with `get_all_embeddings`.
Computes distances to `query_embedding` and maintains the `top_k` closest results, which 
are returned as `(distance, id_text)` tuples in ascending distance order.

# Example

```julia
results = linear_search_all_embeddings("my_database.sqlite", my_query_embedding, "euclidean"; top_k=3)
for (dist, id) in results
    println("ID: ", id, " Distance: ", dist)
end
```

"""function linear_search_all_embeddings(db_path::String, query_embedding::AbstractVector, metric::String; top_k::Int=5)
    all_embs = get_all_embeddings(db_path)
    
    pq = BinaryMaxHeap{Tuple{Float64,String}}()
    for (id, emb) in all_embs
        dist = compute_distance(query_embedding, emb, metric)
        if length(pq) < top_k
            push!(pq, (dist, id))
        elseif dist < peek(pq)[1]
            pop!(pq)
            push!(pq, (dist, id))
        end
    end
    return sort(collect(pq), by=x->x[1])
end



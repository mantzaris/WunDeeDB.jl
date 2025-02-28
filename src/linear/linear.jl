


#TODO: test xxx
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

#TODO: test xxx
function linear_search_ids(db_path::String, query_embedding::AbstractVector, metric::String; top_k::Int=5)
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

#TODO: test xxx
function linear_search_ids_batched(db_path::String, query_embedding::AbstractVector, metric::String;
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


#TODO: test xxx
function linear_search_all_embeddings(db_path::String, query_embedding::AbstractVector, metric::String; top_k::Int=5)
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



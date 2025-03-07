


using SQLite
using DBInterface
using DataFrames
using JSON
using DataStructures: PriorityQueue

# We'll store adjacency in a table "HNSWIndex" with columns:
#   node_id TEXT, layer INTEGER, neighbors TEXT (JSON array of neighbor IDs)
const HNSW_INDEX_TABLE_NAME = "HNSWIndex"


function get_neighbors(db::SQLite.DB, node_id::String)::Dict{Int,Vector{String}}
    
    df = DBInterface.execute(db, 
        "SELECT layer, neighbors 
         FROM $HNSW_INDEX_TABLE_NAME 
         WHERE node_id = ? 
         ORDER BY layer DESC",
        (node_id,)) |> DataFrame

    
    neighbors_dict = Dict{Int,Vector{String}}()
    for row in eachrow(df)
        layer::Int = row.layer
        neighbors_json::String = row.neighbors
        neighbors_array = JSON.parse(neighbors_json)::Vector{String}
        neighbors_dict[layer] = neighbors_array
    end

    return neighbors_dict
end

function save_neighbors(db::SQLite.DB, node_id::String, adjacency::Dict{Int,Vector{String}})
    #remove old adjacency
    DBInterface.execute(db,
        "DELETE FROM $HNSW_INDEX_TABLE_NAME WHERE node_id = ?",
        (node_id,))

    #insert row by row
    for (layer, nbrs) in adjacency
        neighbors_text = JSON.json(nbrs)
        DBInterface.execute(db, """
            INSERT INTO $HNSW_INDEX_TABLE_NAME(node_id, layer, neighbors)
            VALUES (?, ?, ?)
        """, (node_id, layer, neighbors_text))
    end
end


function euclidean_distance(v1::Vector{Float64}, v2::Vector{Float64})
    return sqrt(sum((v1 .- v2) .^ 2))
end

function assign_level(M::Int)::Int
    # random ~ geometric-like distribution
    return floor(Int, -log(rand()) / log(M))
end

function greedy_search_at_level(
    db::SQLite.DB,
    query_vec::Vector{Float64},
    start_id::String,
    level::Int
)::String
    #embed of the start node
    embed_map = get_embeddings(db, [start_id])
    current_vec = embed_map[start_id]
    current_dist = euclidean_distance(query_vec, current_vec)
    current_node_id = start_id

    #adjacency for the current node
    neighbors_dict = get_neighbors(db, current_node_id)

    improved = true
    while improved
        improved = false
        if !haskey(neighbors_dict, level)
            break
        end

        nbrs = neighbors_dict[level]
        nbr_embed_map = get_embeddings(db, nbrs)

        for nbr in nbrs
            d = euclidean_distance(query_vec, nbr_embed_map[nbr])
            if d < current_dist
                current_dist = d
                current_node_id = nbr
                neighbors_dict = get_neighbors(db, current_node_id)
                improved = true
            end
        end
    end
    return current_node_id
end

function search_layer_with_ef(
    db::SQLite.DB,
    query_vec::Vector{Float64},
    start_id::String,
    level::Int,
    ef::Int
)::Vector{String}
    candidate_queue = PriorityQueue{String,Float64}()
    visited = Set{String}()

    #start embedding
    embed_map = get_embeddings(db, [start_id])
    start_vec = embed_map[start_id]
    start_dist = euclidean_distance(query_vec, start_vec)
    candidate_queue[start_id] = start_dist
    push!(visited, start_id)

    top_candidates = PriorityQueue{String,Float64}()
    top_candidates[start_id] = start_dist

    while !isempty(candidate_queue)
        current_id, current_dist = peek(candidate_queue)

        if length(top_candidates) == ef
            worst_id, worst_dist = findmax(top_candidates, by=x->x[2])
            if current_dist >= worst_dist
                break
            end
        end

        pop!(candidate_queue)

        neighbors_dict = get_neighbors(db, current_id)
        if !haskey(neighbors_dict, level)
            continue
        end
        neighbors_list = neighbors_dict[level]
        nbr_embed_map = get_embeddings(db, neighbors_list)

        for nbr in neighbors_list
            if nbr in visited
                continue
            end
            push!(visited, nbr)

            d_nbr = euclidean_distance(query_vec, nbr_embed_map[nbr])

            if length(top_candidates) < ef
                top_candidates[nbr] = d_nbr
                candidate_queue[nbr] = d_nbr
            else
                worst_id, worst_dist = findmax(top_candidates, by=x->x[2])
                if d_nbr < worst_dist
                    delete!(top_candidates, worst_id)
                    top_candidates[nbr] = d_nbr
                    candidate_queue[nbr] = d_nbr
                end
            end
        end
    end

    sorted_tc = sort(collect(top_candidates), by=x->x[2])
    return [tc[1] for tc in sorted_tc]
end

function prune_neighbors(
    db::SQLite.DB,
    new_vec::Vector{Float64},
    candidate_ids::Vector{String},
    M::Int
)::Vector{String}
    if isempty(candidate_ids)
        return String[]
    end
    embed_map = get_embeddings(db, candidate_ids)
    dists = [(cid, euclidean_distance(new_vec, embed_map[cid])) for cid in candidate_ids]
    sorted_cands = sort(dists, by=x->x[2])
    pruned = first(sorted_cands, min(M, length(sorted_cands)))
    return [pc[1] for pc in pruned]
end

function prune_neighbor_list!(
    db::SQLite.DB,
    node_id::String,
    layer::Int,
    M::Int
)
    nbrs_dict = get_neighbors(db, node_id)
    current_neighbors = get(nbrs_dict, layer, String[])

    if length(current_neighbors) <= M
        return
    end

    embed_map = get_embeddings(db, [node_id; current_neighbors])
    node_vec = embed_map[node_id]

    dist_list = [(nbr, euclidean_distance(node_vec, embed_map[nbr])) for nbr in current_neighbors]
    sorted_nbrs = sort(dist_list, by=x->x[2])
    keep = first(sorted_nbrs, M)
    keep_ids = [kn[1] for kn in keep]

    removed = setdiff(Set(current_neighbors), Set(keep_ids))

    nbrs_dict[layer] = keep_ids
    save_neighbors(db, node_id, nbrs_dict)

    # symmetrical unlink
    for r in removed
        r_nbrs_dict = get_neighbors(db, r)
        if haskey(r_nbrs_dict, layer)
            filtered = filter(x-> x != node_id, r_nbrs_dict[layer])
            r_nbrs_dict[layer] = filtered
            save_neighbors(db, r, r_nbrs_dict)
        end
    end
end


function insert!(
    db::SQLite.DB,
    node_id::String;
    M::Int,
    efConstruction::Int,
    efSearch::Int=50,  #might not matter for insertion alone
    entry_point::String="",
    max_level::Int=0
)::Tuple{String, Int}

    embed_map = get_embeddings(db, [node_id])
    if !haskey(embed_map, node_id)
        error("No embedding found for node '$node_id'.")
    end
    new_vec = embed_map[node_id]

    node_level = assign_level(M)
    adjacency = Dict{Int,Vector{String}}()
    for lvl in 0:node_level
        adjacency[lvl] = String[]
    end
    save_neighbors(db, node_id, adjacency)

    local ep = entry_point
    local ml = max_level

    if ep == ""
        # first node
        ep = node_id
        ml = node_level
        return (ep, ml)
    end

    # top-down
    for level in ml:-1:node_level+1
        ep = greedy_search_at_level(db, new_vec, ep, level)
    end

    # BFS
    for level in min(node_level, ml):-1:0
        candidates = search_layer_with_ef(db, new_vec, ep, level, efConstruction)
        pruned = prune_neighbors(db, new_vec, candidates, M)

        # link
        for nbr in pruned
            node_nbrs = get_neighbors(db, node_id)
            push!(node_nbrs[level], nbr)
            save_neighbors(db, node_id, node_nbrs)

            nbr_nbrs = get_neighbors(db, nbr)
            if !haskey(nbr_nbrs, level)
                nbr_nbrs[level] = String[]
            end
            push!(nbr_nbrs[level], node_id)
            save_neighbors(db, nbr, nbr_nbrs)

            prune_neighbor_list!(db, nbr, level, M)
        end

        #prune new node
        prune_neighbor_list!(db, node_id, level, M)

        updated_nbrs = get_neighbors(db, node_id)
        if !isempty(updated_nbrs[level])
            ep = updated_nbrs[level][1]
        end
    end

    if node_level > ml
        ep = node_id
        ml = node_level
    end
    return (ep, ml)
end

function search(
    db::SQLite.DB,
    query_vec::Vector{Float64},
    k::Int;
    efSearch::Int=50,
    entry_point::String="",
    max_level::Int=0
)::Vector{String}
    if entry_point == ""
        return String[]
    end

    current_id = entry_point
    for level in max_level:-1:1
        current_id = greedy_search_at_level(db, query_vec, current_id, level)
    end

    candidates = search_layer_with_ef(db, query_vec, current_id, 0, efSearch)
    if isempty(candidates)
        return String[]
    end

    embed_map = get_embeddings(db, candidates)
    dist_pairs = [(c, euclidean_distance(query_vec, embed_map[c])) for c in candidates]
    dist_sorted = sort(dist_pairs, by=x->x[2])
    best_k = first(dist_sorted, min(k, length(dist_sorted)))
    return [p[1] for p in best_k]
end

function delete!(
    db::SQLite.DB,
    node_id::String;
    entry_point::String="",
    max_level::Int=0
)::Tuple{String, Int}
    nbrs_dict = get_neighbors(db, node_id)
    for (lvl, nbrs) in nbrs_dict
        for nbr in nbrs
            nbr_nbrs = get_neighbors(db, nbr)
            if haskey(nbr_nbrs, lvl)
                filtered = filter(x-> x != node_id, nbr_nbrs[lvl])
                nbr_nbrs[lvl] = filtered
                save_neighbors(db, nbr, nbr_nbrs)
            end
        end
    end

    DBInterface.execute(db,
        "DELETE FROM $HNSW_INDEX_TABLE_NAME WHERE node_id = ?",
        (node_id,))

    local ep = entry_point
    local ml = max_level

    if node_id == ep
        #pick a new entry if any remain
        df = DBInterface.execute(db,
            "SELECT node_id FROM $HNSW_INDEX_TABLE_NAME LIMIT 1") |> DataFrame
        if nrow(df) == 0
            ep = ""
            ml = 0
        else
            new_entry = df[1, "node_id"]
            new_nbrs = get_neighbors(db, new_entry)
            new_ml = isempty(new_nbrs) ? 0 : maximum(keys(new_nbrs))
            ep = new_entry
            ml = new_ml
        end
    end

    return (ep, ml)
end












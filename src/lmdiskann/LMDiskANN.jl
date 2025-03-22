module LMDiskANN

using Random
using SQLite, DataFrames, DBInterface, JSON

import ..WunDeeDB

export search, insert!, delete!


const DEFAULT_MAX_DEGREE = 64  # max number of neighbors
const SEARCH_LIST_SIZE   = 64  # search BFS/greedy queue size
const EF_SEARCH          = 300 # search expansion factor
const EF_CONSTRUCTION    = 400 # construction expansion factor

const LM_DISKANN_CONFIG_TABLE_NAME = "LMDiskANNConfig"
const LM_DISKANN_INDEX_TABLE_NAME = "LMDiskANNIndex"

const INSERT_CONFIG_STMT = """
    INSERT OR REPLACE INTO $(LM_DISKANN_CONFIG_TABLE_NAME) (entrypoint)
    VALUES (?);
"""

const UPDATE_LM_DISKANN_CONFIG_STMT = """
    UPDATE $(LM_DISKANN_CONFIG_TABLE_NAME)
    SET entrypoint = ?;
"""

const GET_CONFIG_STMT = """
    SELECT *
    FROM $(LM_DISKANN_CONFIG_TABLE_NAME)
    LIMIT 1;
"""

const INSERT_LM_DISKANN_NEIGHBORS_STMT = """
    INSERT OR REPLACE INTO $(LM_DISKANN_INDEX_TABLE_NAME) (id_text, neighbors)
    VALUES (?, ?);
"""

const GET_NEIGHBORS_STMT = """
    SELECT neighbors
    FROM $(LM_DISKANN_INDEX_TABLE_NAME)
    WHERE id_text = ?;
"""

const GET_LM_DISKANN_CONFIG_STMT = """
    SELECT *
    FROM $(LM_DISKANN_CONFIG_TABLE_NAME)
    LIMIT 1;
"""







function _get_neighbors(db::SQLite.DB, node_id::String)
    df = DBInterface.execute(db, GET_NEIGHBORS_STMT, (node_id,)) |> DataFrame

    if isempty(df)
        return String[]
    end

    neighbors_str = df[1, :neighbors]
    
    neighbors = JSON.parse(neighbors_str)
    return neighbors
end


function _set_neighbors(db::SQLite.DB, node_id::AbstractString, neighbor_ids::AbstractVector)
    maxd = DEFAULT_MAX_DEGREE

    if length(neighbor_ids) > maxd
        neighbor_ids = neighbor_ids[1:maxd] #update DB TODO: 
    end

    neighbors_str = JSON.json(neighbor_ids)

    DBInterface.execute(db, INSERT_LM_DISKANN_NEIGHBORS_STMT, (node_id, neighbors_str))
end


function _search_graph(db::SQLite.DB, query_vec::AbstractVector, ef::Int)
    df = DBInterface.execute(db, "SELECT entrypoint FROM $(LM_DISKANN_CONFIG_TABLE_NAME) LIMIT 1") |> DataFrame

    if isempty(df)
        return String[]
    end

    entry_id = df[1, :entrypoint]
    if isempty(entry_id)
        return String[]
    end

    entry_dict = WunDeeDB.get_embeddings(db, [entry_id])
    if !haskey(entry_dict, entry_id)
        return String[]
    end

    entry_vec = entry_dict[entry_id]
    entry_dist = WunDeeDB.compute_distance(query_vec, entry_vec, "euclidean")
   
    visited    = Set{String}()
    candidates = Vector{Tuple{Float64,String}}()
    results    = Vector{Tuple{Float64,String}}()

    push!(visited, entry_id)
    push!(candidates, (entry_dist, entry_id))
    push!(results, (entry_dist, entry_id))

    while !isempty(candidates)
        sort!(candidates, by=x->x[1])
        current_dist, current_id = popfirst!(candidates)
        
        if !isempty(results) && last(results)[1] < current_dist
            break
        end
        
        neighbors = _get_neighbors(db, current_id)

        for nbr_id in neighbors
            if nbr_id == "" || (nbr_id in visited)
                continue
            end
            push!(visited, nbr_id)

            nbr_dict = WunDeeDB.get_embeddings(db, [nbr_id])
            if !haskey(nbr_dict, nbr_id)
                continue
            end            

            nbr_vec = nbr_dict[nbr_id]
            nbr_dist= WunDeeDB.compute_distance(query_vec, nbr_vec, "euclidean")

            sort!(results, by=x->x[1])
            if length(results) < ef || nbr_dist < last(results)[1]
                push!(candidates, (nbr_dist, nbr_id))
                push!(results, (nbr_dist, nbr_id))
                if length(results) > ef
                    sort!(results, by=x->x[1])
                    pop!(results)
                end
            end
        end
    end

    sort!(results, by=x->x[1])
    return [pair[2] for pair in results]
end


function search(db::SQLite.DB, query_vec::AbstractVector; topk::Int=10)
    entry_count = WunDeeDB.count_entries(db)
    if entry_count == 0
        return []
    end

    ef_candidates = _search_graph(db, query_vec, max(topk, EF_SEARCH))
    dist_id_pairs = []

    for cid in ef_candidates
        cand_dict = WunDeeDB.get_embeddings(db, [cid])

        if !haskey(cand_dict, cid)
            continue
        end
        
        cand_vec = cand_dict[cid]

        d = WunDeeDB.compute_distance(query_vec, cand_vec, "euclidean")
        push!(dist_id_pairs, (d, cid))
    end

    sort!(dist_id_pairs, by=x->x[1])

    k = min(topk, length(dist_id_pairs))
    results = [pair[2] for pair in dist_id_pairs[1:k]]
    return results
end


function _prune_neighbors(db::SQLite.DB, node_id::AbstractString, candidates::AbstractVector)

    if length(candidates) <= DEFAULT_MAX_DEGREE
        return candidates
    end
    
    node_dict = WunDeeDB.get_embeddings(db, [node_id])
    if !haskey(node_dict, node_id)
        return candidates
    end
        
    node_vec = node_dict[node_id] 
    dist_id_pairs = dist_id_pairs = Vector{Tuple{Float64,String}}()

    for cand_id in candidates        
        cand_dict = WunDeeDB.get_embeddings(db, [cand_id])

        if !haskey(cand_dict, cand_id)
            continue
        end
        
        cand_vec = cand_dict[cand_id]        
        d = WunDeeDB.compute_distance(cand_vec, node_vec, "euclidean")
        push!(dist_id_pairs, (d, cand_id))
    end
    
    sort!(dist_id_pairs, by=x->x[1])
    max_return = min(DEFAULT_MAX_DEGREE, length(dist_id_pairs))
    return [p[2] for p in dist_id_pairs[1:max_return]]
end


function insert!(db::SQLite.DB, node_id::String)
    #SQLite.transaction(db) do
    node_dict = WunDeeDB.get_embeddings(db, [node_id])

    if !haskey(node_dict, node_id)
        error("Node $node_id does not exist in the Embeddings table.")
    end
    
    local_vec = node_dict[node_id]

    df = DBInterface.execute(db, "SELECT entrypoint FROM $(LM_DISKANN_CONFIG_TABLE_NAME) LIMIT 1") |> DataFrame

    if isempty(df)
        DBInterface.execute(db, "INSERT OR REPLACE INTO $(LM_DISKANN_CONFIG_TABLE_NAME) (entrypoint) VALUES (?)", (node_id,) )
        _set_neighbors(db, node_id, String[])
        return
    end

    current_entrypoint = df[1, :entrypoint]

    if length(current_entrypoint) == 0
        DBInterface.execute(db, "UPDATE $(LM_DISKANN_CONFIG_TABLE_NAME) SET entrypoint = ?", (node_id,))
        _set_neighbors(db, node_id, String[])
        return
    end

    # BFS
    nearest_nbrs = search(db, local_vec, topk=DEFAULT_MAX_DEGREE)

    _set_neighbors(db, node_id, nearest_nbrs)

    for nbr_id in nearest_nbrs
        nbr_neighbors = _get_neighbors(db, nbr_id)
        push!(nbr_neighbors, node_id)
        pruned = _prune_neighbors(db, nbr_id, nbr_neighbors)  # prune to max_degree?????!!!!
        _set_neighbors(db, nbr_id, pruned)
    end
    #end
end


function delete!(db::SQLite.DB, node_id::String)

    neighbors = _get_neighbors(db, node_id)

    for nbr_id in neighbors
        nbr_neighbors = _get_neighbors(db, nbr_id)
        filter!(x -> x != node_id, nbr_neighbors)
        _set_neighbors(db, nbr_id, nbr_neighbors)
    end

    _set_neighbors(db, node_id, String[])

    df = DBInterface.execute(db, "SELECT entrypoint FROM $(LM_DISKANN_CONFIG_TABLE_NAME) LIMIT 1") |> DataFrame
    
    if !isempty(df)
        current_entrypoint = df[1, :entrypoint]

        if current_entrypoint == node_id
            df2 = DBInterface.execute(db, "SELECT id_text FROM $(LM_DISKANN_INDEX_TABLE_NAME) WHERE id_text != ? LIMIT 1", (node_id,)) |> DataFrame
            new_entrypoint = isempty(df2) ? "" : df2[1, :id_text]

            DBInterface.execute(db, UPDATE_LM_DISKANN_CONFIG_STMT, (new_entrypoint,))
        end
    end

    DBInterface.execute(db,
        "DELETE FROM $(WunDeeDB.LM_DISKANN_INDEX_TABLE_NAME) WHERE id_text = ?",
        (node_id,)
    )

    return nothing
end

end #end module
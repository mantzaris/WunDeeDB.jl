module LMDiskANN

using Random
import ..WunDeeDB

export build_index, search, insert!, delete!


const DEFAULT_MAX_DEGREE = 64  # max number of neighbors
const SEARCH_LIST_SIZE   = 64  # search BFS/greedy queue size
const EF_SEARCH          = 300 # search expansion factor
const EF_CONSTRUCTION    = 400 # construction expansion factor

const INSERT_CONFIG_STMT = """
    INSERT OR REPLACE INTO $(WunDeeDB.LM_DISKANN_CONFIG_TABLE_NAME) (entrypoint)
    VALUES (?);
"""

const UPDATE_LM_DISKANN_CONFIG_STMT = """
    UPDATE $(WunDeeDB.LM_DISKANN_CONFIG_TABLE_NAME)
    SET entrypoint = ?;
"""

const GET_CONFIG_STMT = """
    SELECT *
    FROM $(WunDeeDB.LM_DISKANN_CONFIG_TABLE_NAME)
    LIMIT 1;
"""

const INSERT_LM_DISKANN_NEIGHBORS_STMT = """
    INSERT OR REPLACE INTO $(WunDeeDB.LM_DISKANN_INDEX_TABLE_NAME) (id_text, neighbors)
    VALUES (?, ?);
"""

const GET_NEIGHBORS_STMT = """
    SELECT neighbors
    FROM $(WunDeeDB.LM_DISKANN_INDEX_TABLE_NAME)
    WHERE id_text = ?;
"""

const GET_LM_DISKANN_CONFIG_STMT = """
    SELECT *
    FROM $(WunDeeDB.LM_DISKANN_CONFIG_TABLE_NAME)
    LIMIT 1;
"""








function _get_neighbors(db::SQLite.DB, node_id::String)
    neighbors = []
    for i in 1:index.maxdegree #TODO: use max degree constant
        nbr_id = index.adjs[i, node_id+1] #TODO: get neighbors from db now
        if nbr_id >= 0
            push!(neighbors, nbr_id)
        end
    end
    return neighbors
end

#TODO: updates the DB
function _set_neighbors(db::SQLite.DB, node_id::String, neighbor_ids::Vector{String})
    maxd = index.maxdegree #TODO: get from db
    if length(neighbor_ids) > maxd
        neighbor_ids = neighbor_ids[1:maxd] #update DB TODO: 
    end

end


function _search_graph(db::SQLite.DB, query_vec::AbstractVector, ef::Int)
    if index.entrypoint < 0 || index.num_points == 0
        return Int[]
    end
    
    visited    = Set()
    candidates = []
    results    = []

    entry_id   = index.entrypoint #TODO: get entry point
    entry_vec  = index.vecs[:, entry_id+1] #get entry vec TODO:
    entry_dist = WunDeeDB.compute_distance(query_vec, entry_vec, "euclidean")

    push!(visited, entry_id)
    push!(candidates, (entry_dist, entry_id))
    push!(results, (entry_dist, entry_id))

    while !isempty(candidates)
        sort!(candidates, by=x->x[1])
        current_dist, current_id = popfirst!(candidates)
        
        if !isempty(results) && last(results)[1] < current_dist
            break
        end
        
        neighbors = _get_neighbors(index, current_id) #TODO: get from db
        for nbr_id in neighbors
            if nbr_id < 0 || (nbr_id in visited)
                continue
            end
            push!(visited, nbr_id)

            nbr_vec = index.vecs[:, nbr_id+1] #TODO: get neighbor embedding
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


function search(db::SQLite.DB, query_vec::AbstractVector; topk::Int=10) #TODO: return only keys not internal / key

    if index.num_points == 0 #TODO: use meta data from db
        return []
    end
    
    ef_candidates = _search_graph(index, query_vec, max(topk, EF_SEARCH))

    dist_id_pairs = []

    for cid in ef_candidates
        v = index.vecs[:, cid+1]
        d = WunDeeDB.compute_distance(query_vec, v, "euclidean")
        push!(dist_id_pairs, (d, cid))
    end
    sort!(dist_id_pairs, by=x->x[1])

    k = min(topk, length(dist_id_pairs))
    results = []

    for i in 1:k
        cid = dist_id_pairs[i][2]
        user_key = get_key_from_id(index.id_mapping_reverse, cid+1)
        push!(results, (user_key, cid+1))
    end
    return results
end


function _prune_neighbors(db::SQLite.DB, node_id::String, candidates::Vector{String})
    maxdegree, indexvecs = getmaxdegreefromdb #TODO:  #TODO: use max degree constant
    if length(candidates) <= index.maxdegree #TODO: use max degree constant
        return candidates
    end
    
    node_vec = index.vecs[:, node_id+1] #give it the index vecs for node TODO: 
    dist_id_pairs = Vector{Tuple{T,Int}}()
    for cand_id in candidates
        cand_vec = index.vecs[:, cand_id+1]
        d = WunDeeDB.compute_distance(cand_vec, node_vec, "euclidean")
        push!(dist_id_pairs, (d, cand_id))
    end
    sort!(dist_id_pairs, by=x->x[1])
    return [p[2] for p in dist_id_pairs[1:index.maxdegree]]
end


function insert!(db::SQLite.DB, node_id::String)
    #SQLite.transaction(db) do
    node_dict = WunDeeDB.get_embeddings(db, [node_id])

    if !haskey(node_dict, node_id)
        error("Node $node_id does not exist in the Embeddings table.")
    end
    
    local_vec = node_dict[node_id]

    df = DBInterface.execute(db, "SELECT entrypoint FROM $(WunDeeDB.LM_DISKANN_CONFIG_TABLE_NAME) LIMIT 1") |> DataFrame

    if isempty(df)
        DBInterface.execute(db, "INSERT OR REPLACE INTO $(WunDeeDB.LM_DISKANN_CONFIG_TABLE_NAME) (entrypoint) VALUES (?)", (node_id,) )
        _set_neighbors(db, node_id, String[])
        return
    end

    current_entrypoint = df[1, :entrypoint]

    if length(current_entrypoint) == 0
        DBInterface.execute(db, "UPDATE $(WunDeeDB.LM_DISKANN_CONFIG_TABLE_NAME) SET entrypoint = ?", (node_id,))
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

    df = DBInterface.execute(db, "SELECT entrypoint FROM $(WunDeeDB.LM_DISKANN_CONFIG_TABLE_NAME) LIMIT 1") |> DataFrame
    
    if !isempty(df)
        current_entrypoint = df[1, :entrypoint]

        if current_entrypoint == node_id
            df2 = DBInterface.execute(db, "SELECT id_text FROM $(WunDeeDB.LM_DISKANN_INDEX_TABLE_NAME) WHERE id_text != ? LIMIT 1", (node_id,)) |> DataFrame
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
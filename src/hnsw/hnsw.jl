
using DataStructures: PriorityQueue #TODO: remove as not needed when used from WunDeeDB file

#data structures
struct HNSWNode
    id::Int
    vector::Vector{Float64}
    level::Int
    neighbors::Dict{Int, Vector{Int}}
end

struct HNSW
    nodes::Dict{Int, HNSWNode}
    entry_point::Int
    max_level::Int
    M::Int
    efConstruction::Int
    efSearch::Int
end

#constructor
function create_hnsw(M::Int; efConstruction::Int=200, efSearch::Int=50)
    return HNSW(Dict(), -1, 0, M, efConstruction, efSearch)
end

#distance
function euclidean_distance(v1::Vector{Float64}, v2::Vector{Float64})
    return sqrt(sum((v1 .- v2) .^ 2))
end

#random level assignment
function assign_level(M::Int)::Int
    return floor(Int, -log(rand()) / log(M))
end

#greedy search at level (unchanged from your version)
function greedy_search_at_level(index::HNSW, query::Vector{Float64}, start_id::Int, level::Int)
    current_node = index.nodes[start_id]
    current_dist = euclidean_distance(query, current_node.vector)
    improved = true
    while improved
        improved = false
        for neighbor_id in current_node.neighbors[level]
            neighbor_node = index.nodes[neighbor_id]
            d = euclidean_distance(query, neighbor_node.vector)
            if d < current_dist
                current_node = neighbor_node
                current_dist = d
                improved = true
            end
        end
    end
    return current_node.id
end




#extended BFS for a given level with a pool size = ef
function search_layer_with_ef(
    index::HNSW,
    query::Vector{Float64},
    start_id::Int,
    level::Int,
    ef::Int
)
    #store distances in the priority queue so the 'lowest distance' is always popped first by default
    candidate_queue = PriorityQueue{Int, Float64}()
    visited = Set{Int}()

    start_dist = euclidean_distance(query, index.nodes[start_id].vector)
    candidate_queue[start_id] = start_dist
    push!(visited, start_id)

    #keep top-candidates (up to 'ef') in a separate PriorityQueue
    top_candidates = PriorityQueue{Int, Float64}()
    top_candidates[start_id] = start_dist

    while !isempty(candidate_queue)
        #look at the best candidate in the queue
        current_id, current_dist = peek(candidate_queue)

        #early-exit check: if top_candidates is full AND
        #best candidate in the queue is not better than the worst in top_candidates, stop
        if length(top_candidates) == ef
            worst_id, worst_dist = findmax(top_candidates, by=x->x[2])
            if current_dist >= worst_dist
                #if the queue's best is still worse than or equal to the worst in top_candidates
                #no further improvement is possible, break early
                break
            end
        end

        #actually dequeue the item so we can expand neighbors
        pop!(candidate_queue)

        #expand neighbors
        for neighbor_id in index.nodes[current_id].neighbors[level]
            if neighbor_id in visited
                continue
            end
            push!(visited, neighbor_id)

            d_neighbor = euclidean_distance(query, index.nodes[neighbor_id].vector)

            #if top_candidates is not full OR we found a better candidate than the worst
            if length(top_candidates) < ef
                top_candidates[neighbor_id] = d_neighbor
                candidate_queue[neighbor_id] = d_neighbor
            else
                worst_id, worst_dist = findmax(top_candidates, by=x->x[2])
                if d_neighbor < worst_dist
                    #remove the worst from top_candidates
                    delete!(top_candidates, worst_id)
                    top_candidates[neighbor_id] = d_neighbor
                    candidate_queue[neighbor_id] = d_neighbor
                end
            end
        end
    end

    #finally, return the IDs in top_candidates, sorted by ascending distance
    sorted_tc = sort(collect(top_candidates), by = x -> x[2])
    return [tc[1] for tc in sorted_tc]
end



#prune neighbor candidates to top M closest to new_node
function prune_neighbors(index::HNSW,
                         new_node::HNSWNode,
                         candidates::Vector{Int},
                         level::Int,
                         M::Int)
    dists = [(cand_id, euclidean_distance(new_node.vector, index.nodes[cand_id].vector))
             for cand_id in candidates]
    sorted_cands = sort(dists, by=x->x[2])  # ascending distance
    pruned = first(sorted_cands, min(M, length(sorted_cands)))
    return [pc[1] for pc in pruned]
end


#prune a single neighbor’s adjacency list to max M
#remove symmetrical references from any neighbor that got dropped
function prune_neighbor_list!(
    index::HNSW,
    neighbor_id::Int,
    level::Int,
    M::Int
)
    neighbor_node = index.nodes[neighbor_id]
    current_neighbors = neighbor_node.neighbors[level]
    
    #if already within M, do nothing
    if length(current_neighbors) <= M
        return
    end

    #sort neighbors by their distance to this node
    distances = [(nbr, euclidean_distance(index.nodes[nbr].vector, neighbor_node.vector))
                 for nbr in current_neighbors]
    sorted_nbrs = sort(distances, by = x -> x[2])   #ascending order
    keep = first(sorted_nbrs, M)  #keep up to M
    keep_ids = [kn[1] for kn in keep]

    #figure out which was removed
    old_set = Set(current_neighbors)
    new_set = Set(keep_ids)
    removed = setdiff(old_set, new_set)

    # Update this node's adjacency list
    neighbor_node.neighbors[level] = keep_ids

    #symmetrical un-linking: For each removed neighbor r
    #remove neighbor_id from r's adjacency list
    for r in removed
        other_node = index.nodes[r]
        deleteat!(
            other_node.neighbors[level],
            findall(==(neighbor_id), other_node.neighbors[level])
        )
    end
end


function insert!(index::HNSW, id::Int, vector::Vector{Float64})
    node_level = assign_level(index.M)
    new_node = HNSWNode(id, vector, node_level, Dict{Int, Vector{Int}}())
    for level in 0:node_level
        new_node.neighbors[level] = Int[]
    end

    if isempty(index.nodes)
        index.nodes[id] = new_node
        index.entry_point = id
        index.max_level = node_level
        return
    end

    #insert the new_node into the dictionary NOW, so symmetrical operations work
    index.nodes[id] = new_node

    #greedy top-down
    current_id = index.entry_point
    for level in index.max_level:-1:node_level+1
        current_id = greedy_search_at_level(index, vector, current_id, level)
    end

    #BFS + pruning from min(node_level, max_level) down to 0
    for level in min(node_level, index.max_level):-1:0
        #search
        candidates = search_layer_with_ef(index, vector, current_id, level, index.efConstruction)

        #prune to M
        pruned_neighbors = prune_neighbors(index, new_node, candidates, level, index.M)

        #link
        for neighbor_id in pruned_neighbors
            push!(new_node.neighbors[level], neighbor_id)
            push!(index.nodes[neighbor_id].neighbors[level], new_node.id)
            prune_neighbor_list!(index, neighbor_id, level, index.M)
        end

        #prune new node's adjacency, now new_node is in index
        prune_neighbor_list!(index, new_node.id, level, index.M)

        #pick closest neighbor to continue top-down
        if !isempty(pruned_neighbors)
            current_id = pruned_neighbors[1]
        end
    end

    if node_level > index.max_level
        index.entry_point = id
        index.max_level = node_level
    end
end




function search(index::HNSW, query::Vector{Float64}, k::Int)

    current_id = index.entry_point
    for level in index.max_level:-1:1
        current_id = greedy_search_at_level(index, query, current_id, level)
    end

    #BFS seaerch
    candidates = search_layer_with_ef(index, query, current_id, 0, index.efSearch)

    sorted_candidates = sort(candidates, by = c -> euclidean_distance(query, index.nodes[c].vector))
    return sorted_candidates[1:min(k, length(sorted_candidates))]
end

#extendded HNSW might do more advanced re-linking TODO:
function delete!(index::HNSW, id::Int)
    if !haskey(index.nodes, id)
        error("Node $id not found.")
    end
    node = index.nodes[id]
    for (level, nbrs) in node.neighbors
        for nbr in nbrs
            neighbor_node = index.nodes[nbr]
            deleteat!(neighbor_node.neighbors[level], findall(==(id), neighbor_node.neighbors[level]))
        end
    end
    delete!(index.nodes, id)
    
    if id == index.entry_point
        if !isempty(index.nodes)
            new_entry = findmax([(n.level, nid) for (nid,n) in index.nodes])[2]
            index.entry_point = new_entry
            index.max_level = index.nodes[new_entry].level
        else
            index.entry_point = -1
            index.max_level = 0
        end
    end
end


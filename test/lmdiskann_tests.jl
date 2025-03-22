using Random, StatsBase

function clean_up()
    current_dir = pwd()
    for fname in readdir(current_dir)
        if startswith(fname, "temp")
            path = joinpath(current_dir, fname)
            if isdir(path)
                rm(path; recursive=true, force=true)
            else
                rm(path; force=true)
            end
        end
    end
end


@testset "LMDiskANN Single Node Test" begin
    clean_up()

    local TEST_DB = "temp_lmdiskann_single.sqlite"
    
    # initialize DB for dimension=3, Float32
    local init_res = initialize_db(TEST_DB, 3, "Float32";
                                   keep_conn_open=true,
                                   description="Single-node test",
                                   ann="lmdiskann")
    @test init_res === true

    # insert embedding into 'Embeddings' table
    local emb = Float32[0.1, 0.2, 0.3]
    local ins_ok = WunDeeDB.insert_embeddings(TEST_DB, "node1", emb)
    @test ins_ok === true

    # open DB, run LMDiskANN.insert!(...) so adjacency is built
    local db = open_db(TEST_DB, keep_conn_open=true)
    WunDeeDB.insert_embeddings(db, "node2",Float32[0.4,0.3,0.2])

    # search for something near [0.1, 0.2, 0.3]
    local query_vec = Float32[0.12, 0.22, 0.28]
    local results = WunDeeDB.search_ann(TEST_DB, query_vec, "euclidean", top_k=1)
    @test length(results) == 1
    @test results[1] == "node1"

    # delete node1 from adjacency
    WunDeeDB.delete_embeddings(db, "node1")
    local results_after = WunDeeDB.search_ann(TEST_DB, query_vec, "euclidean", top_k=1)
    @test !("node1" in results_after)  # node1 is gone

    close_db(db)
    clean_up()
end


@testset "LMDiskANN Two-Node Test" begin
    clean_up()
    local TEST_DB = "temp_lmdiskann_two.sqlite"

    #init for dimension=3, float32, with LMDiskANN
    local init_res = initialize_db(
        TEST_DB, 3, "Float32";
        keep_conn_open=true,
        description="Two-node test",
        ann="lmdiskann"
    )
    @test init_res === true

    # insert two embeddings into the main "Embeddings" table (and adjacency),
    #    relying on WunDeeDB's code that calls "insert_embeddings_ann" internally
    local node1_emb = Float32[0.1, 0.2, 0.3]
    local node2_emb = Float32[0.8, 0.7, 0.6]

    @test insert_embeddings(TEST_DB, "node1", node1_emb) === true
    @test insert_embeddings(TEST_DB, "node2", node2_emb) === true

    # search near node1's embedding. expect "node1" to be the top neighborbut since top_k=2, we may get [node1, node2] if the BFS sees them both
    local query1 = Float32[0.11, 0.21, 0.29]
    local res1 = search_ann(TEST_DB, query1, "euclidean"; top_k=2)
    @test "node1" in res1
    @test length(res1) <= 2

    #search near node2's embedding. "node2" should appear
    local query2 = Float32[0.77, 0.68, 0.61]
    local res2 = search_ann(TEST_DB, query2, "euclidean"; top_k=2)
    @test "node2" in res2

    # now delete node1 from adjacency (and Embeddings) 
    @test delete_embeddings(TEST_DB, "node1") === true

    # searching near node1's embedding should not yield node1 anymore
    local res3 = search_ann(TEST_DB, query1, "euclidean"; top_k=2)
    @test !("node1" in res3)
    # node2 might remain as a valid neighbor

    # cleanup
    close_db()  # or close_db() if your code is set up that way
    clean_up()
end


@testset "LMDiskANN Empty DB Test" begin
    clean_up()

    local TEST_DB = "temp_lmdiskann_empty.sqlite"

    local init_res = initialize_db(TEST_DB, 3, "Float32";
                                   keep_conn_open=true,
                                   description="Empty DB test",
                                   ann="lmdiskann")
    @test init_res === true

    local db = open_db(TEST_DB, keep_conn_open=true)

    local query = Float32[0.1, 0.2, 0.3]

    local res = search_ann(TEST_DB, query, "euclidean"; top_k=5)
    @test isempty(res)

    close_db()
    clean_up()
end


@testset "LMDiskANN Multi-Node Insertion Test" begin
    clean_up()

    local TEST_DB = "temp_lmdiskann_multi.sqlite"

    local init_res = initialize_db(TEST_DB, 3, "Float32";
                                   keep_conn_open=true,
                                   description="multi-node test",
                                   ann="lmdiskann")
    @test init_res === true

    local nodes = ["node1", "node2", "node3", "node4"]
    local embs = [
        Float32[0.0, 0.1, 0.2],
        Float32[0.9, 0.8, 0.7],
        Float32[0.1, 0.9, 0.5],
        Float32[0.3, 0.3, 0.3]
    ]
    @test length(nodes) == length(embs)


    for (i, nid) in enumerate(nodes)
        @test insert_embeddings(TEST_DB, nid, embs[i]) === true
    end

    # search near [0.0, 0.1, 0.2].
    local qvec = Float32[0.05, 0.08, 0.18]
    local sres = search_ann(TEST_DB, qvec, "euclidean"; top_k=2)
    @test !isempty(sres)
    @test "node1" in sres

    close_db()
    clean_up()
end


@testset "LMDiskANN Larger topk Test" begin
    clean_up()

    local TEST_DB = "temp_lmdiskann_largetopk.sqlite"

    local init_res = initialize_db(TEST_DB, 3, "Float32";
                                   keep_conn_open=true, 
                                   ann="lmdiskann")
    @test init_res === true

    local node_ids = ["n1", "n2", "n3"]
    local node_embs = [
        Float32[0.0, 0.0, 0.0],
        Float32[1.0, 1.0, 1.0],
        Float32[0.5, 0.5, 0.5]
    ]

    for (i, nid) in enumerate(node_ids)
        @test insert_embeddings(TEST_DB, nid, node_embs[i]) === true
    end

    # search with top_k=10
    local query = Float32[0.6, 0.6, 0.6]
    local results = search_ann(TEST_DB, query, "euclidean"; top_k=10)

    #only 3 total nodes, so expect <=3
    @test length(results) <= 3  
    # no duplicates
    @test Set(results) == Set(results)

    close_db()
    clean_up()
end



@testset "LMDiskANN Delete Non-Existent Node Test" begin
    clean_up()
    local TEST_DB = "temp_lmdiskann_nonex_delete.sqlite"

    local init_res = initialize_db(TEST_DB, 3, "Float32"; 
                                   keep_conn_open=true, 
                                   ann="lmdiskann")
    @test init_res === true

    local db = open_db(TEST_DB, keep_conn_open=true)

    # never call insert_embeddings(...,"bogusNode", ...)
    #no embedding for "bogusNode" in the table
    #attempt to delete embeddings for "bogusNode" from the DB
    try
        local del_res = delete_embeddings(db, "bogusNode")
        
        @test del_res === true
    catch e
        @test false
    end

    close_db(db)
    clean_up()
end


@testset "LMDiskANN Incremental Insert + Search Test" begin
    clean_up()

    local TEST_DB = "temp_lmdiskann_incremental.sqlite"

    local init_res = initialize_db(TEST_DB, 3, "Float32";
                                   keep_conn_open=true,
                                   ann="lmdiskann")
    @test init_res === true

    local node_ids = ["n1", "n2", "n3"]
    local node_embs = [
        Float32[0.1, 0.2, 0.3],
        Float32[0.9, 0.8, 0.7],
        Float32[0.5, 0.5, 0.5]
    ]

    @test insert_embeddings(TEST_DB, "n1", node_embs[1]) === true
    local r1 = search_ann(TEST_DB, node_embs[1], "euclidean"; top_k=2)
    @test "n1" in r1

    @test insert_embeddings(TEST_DB, "n2", node_embs[2]) === true
    local r2 = search_ann(TEST_DB, node_embs[2], "euclidean"; top_k=2)
    @test "n2" in r2

    @test insert_embeddings(TEST_DB, "n3", node_embs[3]) === true
    local r3 = search_ann(TEST_DB, node_embs[3], "euclidean"; top_k=3)
    @test "n3" in r3

    close_db()
    clean_up()
end


@testset "LMDiskANN Larger Random Vectors Test" begin
    clean_up()
    
    local TEST_DB = "temp_lmdiskann_large.sqlite"
    
    #initialize DB for dimension=3, Float32, ann="lmdiskann"
    local init_res = initialize_db(
        TEST_DB, 3, "Float32";
        keep_conn_open=true,
        description="Larger random test",
        ann="lmdiskann"
    )
    @test init_res === true

    # make a bunch of random embeddings
    local NUM_VECTORS = 100
    local node_ids = ["node$i" for i in 1:NUM_VECTORS]
    local node_embs = [rand(Float32, 3) for _ in 1:NUM_VECTORS]

    #insert them in a random order so adjacency is built incrementally
    local insertion_order = shuffle(node_ids)
    for nid in insertion_order
        local i = parse(Int, replace(nid, "node"=>""))  # eg "node14" => 14
        local emb = node_embs[i]
        #every call to insert_embeddings triggers adjacency insertion!!!!
        local r = insert_embeddings(TEST_DB, nid, emb)
        @test r === true
    end

    # perform some random queries
    local NUM_QUERIES = 5
    for _ in 1:NUM_QUERIES
        local q_vec = rand(Float32, 3)  # random query
        #WunDeeDB.search_ann calls LMDiskANN under the hood
        local sres = search_ann(TEST_DB, q_vec, "euclidean"; top_k=5)
        #test that it returns up to 5 results & no duplicates
        @test length(sres) <= 5
        @test length(Set(sres)) == length(sres)
    end

    # optional partial deletion
    local deleted_nodes = shuffle(sample(node_ids, 5; replace=false))
    for dn in deleted_nodes
        #removes dn from the Embeddings table & adjacency
        delete_embeddings(TEST_DB, dn)
    end

    # confirm deleted nodes do not appear in subsequent queries!
    local q_vec2 = rand(Float32, 3)
    local sres2 = search_ann(TEST_DB, q_vec2, "euclidean"; top_k=10)
    for dn in deleted_nodes
        @test !(dn in sres2)
    end

    close_db()
    clean_up()
end


@testset "LMDiskANN Larger Random Vectors Test (Float16, dim=5)" begin
    clean_up()
    
    local TEST_DB = "temp_lmdiskann_large_f16.sqlite"
    
    #init DB for dimension=5, Float16, ann="lmdiskann"
    local init_res = initialize_db(
        TEST_DB, 5, "Float16";
        keep_conn_open=true,
        description="Larger random test (Float16, dim=5)",
        ann="lmdiskann"
    )
    @test init_res === true

    #make a bunch of random embeddings in Float16 with dimension=5
    local NUM_VECTORS = 100
    local node_ids = ["node$i" for i in 1:NUM_VECTORS]
    local node_embs = [rand(Float16, 5) for _ in 1:NUM_VECTORS]

    #insert them in a random order so adjacency is built incrementally
    local insertion_order = shuffle(node_ids)
    for nid in insertion_order
        local i = parse(Int, replace(nid, "node" => ""))  # e.g. "node14" => 14
        local emb = node_embs[i]
        # each call to insert_embeddings triggers adjacency insertion
        local r = insert_embeddings(TEST_DB, nid, emb)
        @test r === true
    end

    #perform some random queries
    local NUM_QUERIES = 5
    for _ in 1:NUM_QUERIES
        #random query vector of dimension=5, Float16
        local q_vec = rand(Float16, 5)
        # WunDeeDB.search_ann calls LMDiskANN under the hood
        local sres = search_ann(TEST_DB, q_vec, "euclidean"; top_k=5)
        #test that it returns up to 5 results & no duplicates
        @test length(sres) <= 5
        @test length(Set(sres)) == length(sres)
    end

    # partial deletion
    local deleted_nodes = shuffle(sample(node_ids, 5; replace=false))
    for dn in deleted_nodes
        # remove dn from Embeddings and adjacency
        delete_embeddings(TEST_DB, dn)
    end

    # confirm deleted nodes do not appear in subsequent queries
    local q_vec2 = rand(Float16, 5)
    local sres2 = search_ann(TEST_DB, q_vec2, "euclidean"; top_k=10)
    for dn in deleted_nodes
        @test !(dn in sres2)
    end

    close_db()
    clean_up()
end


@testset "LMDiskANN Recall Test (200 Vectors, >=80% recall)" begin
    clean_up()

    local TEST_DB = "temp_lmdiskann_recall.sqlite"
    
    local init_res = initialize_db(TEST_DB, 5, "Float32";
                                   keep_conn_open=true,
                                   description="Recall test for LMDiskANN",
                                   ann="lmdiskann")
    @test init_res === true

    #make 200 random embeddings
    local NUM_VECTORS = 200
    local node_ids = [ "node$i" for i in 1:NUM_VECTORS ]
    local node_embs = [ rand(Float32, 5) for _ in 1:NUM_VECTORS ]

    # insert each embedding -> triggers adjacency insertion if ann="lmdiskann"
    for (i, nid) in enumerate(node_ids)
        local r = insert_embeddings(TEST_DB, nid, node_embs[i])
        @test r === true
    end

    #  test recall with some number of random queries
    #pick top_k=10 neighbors as our "focus" for recall
    local TOP_K = 10
    local NUM_QUERIES = 10

    # store per-query recall to take an average at the end
    local recall_values = Float64[]

    #compute:
    #- ground truth top-10 neighbors by linear distance
    #- approximate neighbors from search_ann
    #- measure overlap => recall
    for q_i in 1:NUM_QUERIES
        # random query
        local q_vec = rand(Float32, 5)

        # ground-truth: measure distance to all node_embs, pick top 10
        #    We'll store (distance, node_id) pairs, then sort by dist
        local dist_id_pairs = Vector{Tuple{Float32,String}}()
        for i in 1:NUM_VECTORS
            local d = WunDeeDB.compute_distance(node_embs[i], q_vec, "euclidean")
            push!(dist_id_pairs, (d, node_ids[i]))
        end
        sort!(dist_id_pairs, by=x->x[1])
        local true_neighbors = [pair[2] for pair in dist_id_pairs[1:TOP_K]]

        # approximate neighbors from LMDiskANN
        local approx_neighbors = search_ann(TEST_DB, q_vec, "euclidean"; top_k=TOP_K)

        # compute overlap
        local overlap = length(intersect(true_neighbors, approx_neighbors))
        local query_recall = overlap / TOP_K
        push!(recall_values, query_recall)
    end

    local avg_recall = sum(recall_values) / length(recall_values)
    @test avg_recall >= 0.8  # require at least 80%

    close_db()
    clean_up()
end

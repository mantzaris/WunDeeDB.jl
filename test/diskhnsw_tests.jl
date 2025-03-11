

@testset "diskhnsw tests" begin
    println("XXXXXXXXXXXXXXXXXXX")
    @test 1 == 1

end




@testset "DiskHNSW Basic Tests" begin
    local TEST_DB = "temp_diskhnsw_integration.sqlite"
    for f in readdir(".")
        if startswith(f, "temp_diskhnsw_integration")
            rm(f; force=true)
        end
    end

    local init_res = initialize_db(TEST_DB, 3, "Float32"; keep_conn_open=true, description="DiskHNSW test DB", ann="hnsw")
    @test init_res === true

    local emb1 = Float32[0.1, 0.2, 0.3]
    local emb2 = Float32[0.9, 0.8, 0.7]

    local ins1 = insert_embeddings(TEST_DB, "node1", emb1)
    @test ins1 === true

    local ins2 = insert_embeddings(TEST_DB, "node2", emb2)
    @test ins2 === true

    local db = open_db(TEST_DB, keep_conn_open=true)

    #M=4, efConstruction=20 
    local ep, ml = WunDeeDB.DiskHNSW.insert!(db, "node1"; M=4, efConstruction=20, efSearch=50, entry_point="", max_level=0)
    @test ep == "node1"  
    @test ml >= 0

    
    local new_ep, new_ml = WunDeeDB.DiskHNSW.insert!(db, "node2"; M=4, efConstruction=20, efSearch=50, entry_point=ep, max_level=ml)
    @test new_ep !== "" 

    local query_vec = Float32[1.0, 2.0, 3.0]
    local results = WunDeeDB.DiskHNSW.search(db, query_vec, 2; efSearch=50, entry_point=new_ep, max_level=new_ml)
    @test length(results) > 0
    @test "node2" in results 


    local del_ep, del_ml = WunDeeDB.DiskHNSW.delete!(db, "node1"; entry_point=new_ep, max_level=new_ml)

    local results2 = WunDeeDB.DiskHNSW.search(db, query_vec, 2; efSearch=50, entry_point=del_ep, max_level=del_ml)
    @test "node1" ∉ results2

    close_db(db)
    rm(TEST_DB; force=true)  
end



@testset "DiskHNSW Multi-Node Tests" begin
    # 1) Remove old DB
    local TEST_DB = "temp_diskhnsw_multi.sqlite"
    for f in readdir(".")
        if startswith(f, "temp_diskhnsw_multi")
            rm(f; force=true)
        end
    end

    # 2) Initialize fresh DB with ann="hnsw"
    local init_res = initialize_db(TEST_DB, 3, "Float32"; keep_conn_open=true, description="multi-node test", ann="hnsw")
    @test init_res === true

    # 3) Prepare 5–6 embeddings 
    local node_ids = ["node$i" for i in 1:5]
    local node_embs = [
        Float32[0.1, 0.2, 0.3],
        Float32[1.0, 2.0, 3.0],
        Float32[0.5, 0.5, 0.5],
        Float32[3.1, 1.59, 2.65],
        Float32[9.99, 9.88, 9.77],
    ]

    # Insert them into the main table
    for (i, nid) in enumerate(node_ids)
        local res_ins = insert_embeddings(TEST_DB, nid, node_embs[i])
        @test res_ins === true
    end

    # 4) Open a connection to do HNSW insertion
    local db = open_db(TEST_DB, keep_conn_open=true)

    # We'll keep track of the current entry point
    local ep = ""
    local ml = 0

    # Insert each node in a random order 
    local insertion_order = shuffle(node_ids)
    for nid in insertion_order
        ep, ml = WunDeeDB.DiskHNSW.insert!(
            db, nid;
            M=4, efConstruction=10, efSearch=50,  # example
            entry_point=ep, max_level=ml
        )
    end

    @test ep != ""
    @test ml >= 0

    # 5) Do a few searches 
    #    e.g., pick a vector near [1.0,2.0,3.0] => node2 should appear
    local query_vec = Float32[0.9, 1.9, 3.1]
    local results = WunDeeDB.DiskHNSW.search(db, query_vec, 3; efSearch=15, entry_point=ep, max_level=ml)
    @test length(results) <= 3
    @test "node2" in results  # likely

    # Another query near [0.1, 0.2, 0.3]
    local q2 = Float32[0.15, 0.25, 0.28]
    local r2 = WunDeeDB.DiskHNSW.search(db, q2, 2; efSearch=8, entry_point=ep, max_level=ml)
    @test length(r2) <= 2
    @test "node1" in r2  # or whichever

    # 6) Test BFS or prune logic by adding a new node with a big random level
    local emb6 = Float32[2.0, 2.0, 2.0]
    local ins6 = insert_embeddings(TEST_DB, "node6", emb6)
    @test ins6 === true
    local ep2, ml2 = WunDeeDB.DiskHNSW.insert!(
        db, "node6"; M=4, efConstruction=10, efSearch=50,
        entry_point=ep, max_level=ml
    )
    # might become the new entry point if it gets a higher assigned level
    @test (ep2 != "") && (ml2 >= ml)
    
    # 7) Another search after node6 inserted
    local r3 = WunDeeDB.DiskHNSW.search(db, Float32[2.05,2.05,2.05], 3; efSearch=12, entry_point=ep2, max_level=ml2)
    @test "node6" in r3
    
    # 8) Optionally test deletion of "node2" 
    local del_ep, del_ml = WunDeeDB.DiskHNSW.delete!(db, "node2"; entry_point=ep2, max_level=ml2)
    @test "node2" ∉ WunDeeDB.DiskHNSW.search(db, query_vec, 3; efSearch=12, entry_point=del_ep, max_level=del_ml)
    
    close_db(db)
    rm(TEST_DB; force=true)
end

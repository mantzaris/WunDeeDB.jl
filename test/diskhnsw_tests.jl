

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

    # M=4, efConstruction=20 just as an example
    local ep, ml = WunDeeDB.DiskHNSW.insert!(db, "node1"; M=4, efConstruction=20, efSearch=50, entry_point="", max_level=0)
    @test ep == "node1"  
    @test ml >= 0

    # Insert the second node. Now we pass the existing ep, ml
    local new_ep, new_ml = WunDeeDB.DiskHNSW.insert!(db, "node2"; M=4, efConstruction=20, efSearch=50, entry_point=ep, max_level=ml)
    @test new_ep !== ""  # might remain "node1" or change to "node2" if the second node has a higher level

    local query_vec = Float32[1.0, 2.0, 3.0]
    local results = WunDeeDB.DiskHNSW.search(db, query_vec, 2; efSearch=50, entry_point=new_ep, max_level=new_ml)
    @test length(results) > 0
    @test "node2" in results   # node2 should appear if it's the nearest

    # 6) Optionally test 'delete!' logic
    local del_ep, del_ml = WunDeeDB.DiskHNSW.delete!(db, "node1"; entry_point=new_ep, max_level=new_ml)

    local results2 = WunDeeDB.DiskHNSW.search(db, query_vec, 2; efSearch=50, entry_point=del_ep, max_level=del_ml)
    @test "node1" ∉ results2

    close_db(db)
    rm(TEST_DB; force=true)  # remove if you want a fully ephemeral test
end


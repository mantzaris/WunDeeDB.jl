

@testset "diskhnsw tests" begin
    println("XXXXXXXXXXXXXXXXXXX")
    @test 1 == 1

end



@testset "DiskHNSW Basic Tests (Automatic Insertion)" begin
    local TEST_DB = "temp_diskhnsw_test.sqlite"
    # Clean up any existing file
    if isfile(TEST_DB)
        rm(TEST_DB; force=true)
    end

    # 1) Initialize the DB with ann="hnsw"
    #    This creates the HNSW index/config tables and inserts defaults if empty.
    res_init = initialize_db(TEST_DB, 3, "Float32"; ann="hnsw", keep_conn_open=true)
    @test res_init === true

    # 2) Open a persistent connection
    db = open_db(TEST_DB, keep_conn_open=true)

#     # 3) Insert some embeddings => automatically also inserted into HNSW
    local emb1 = Float32[0.1, 0.2, 0.3]
    local emb2 = Float32[0.9, 0.8, 0.7]

    @test insert_embeddings(db, "node1", emb1) === true
    @test insert_embeddings(db, "node2", emb2) === true
    # # Because ann="hnsw", your code automatically calls DiskHNSW.insert! behind the scenes.

    # # 4) Search for the node(s)
    # local query = Float32[0.05, 0.15, 0.25]
    # local results = search_ann(TEST_DB, query, "euclidean"; top_k=2)
    # @test length(results) > 0
    # @test "node1" in results

    # # 5) Delete node1 => the HNSW index is also updated automatically
    # delete_embeddings(db, "node1")

    # # 6) Repeat the same search => node1 should no longer appear
    # local results2 = search_ann(TEST_DB, query, "euclidean"; top_k=2)
    # @test !("node1" in results2)

    # # 7) Clean up: close DB, remove file
    close_db(db)
    if isfile(TEST_DB)
        rm(TEST_DB; force=true)
    end
end

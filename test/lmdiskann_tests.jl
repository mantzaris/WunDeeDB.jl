

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
    local ins_ok = insert_embeddings(TEST_DB, "node1", emb)
    @test ins_ok === true

    # open DB, run LMDiskANN.insert!(...) so adjacency is built
    local db = open_db(TEST_DB, keep_conn_open=true)
    WunDeeDB.LMDiskANN.insert!(db, "node1")

    # search for something near [0.1, 0.2, 0.3]
    local query_vec = Float32[0.12, 0.22, 0.28]
    local results = WunDeeDB.LMDiskANN.search(db, query_vec, topk=1)
    @test length(results) == 1
    @test results[1] == "node1"

    # delete node1 from adjacency
    WunDeeDB.LMDiskANN.delete!(db, "node1")
    local results_after = WunDeeDB.LMDiskANN.search(db, query_vec, topk=1)
    @test !("node1" in results_after)  # node1 is gone

    close_db(db)
    clean_up()
end


@testset "LMDiskANN Two-Node Test" begin
    clean_up()
    local TEST_DB = "temp_lmdiskann_two.sqlite"

    local init_res = initialize_db(TEST_DB, 3, "Float32";
                                   keep_conn_open=true,
                                   description="Two-node test",
                                   ann="lmdiskann")
    @test init_res === true

    local node1_emb = Float32[0.1, 0.2, 0.3]
    local node2_emb = Float32[0.8, 0.7, 0.6]

    @test insert_embeddings(TEST_DB, "node1", node1_emb) === true
    @test insert_embeddings(TEST_DB, "node2", node2_emb) === true

    local db = open_db(TEST_DB, keep_conn_open=true)

    # insert node1
    WunDeeDB.LMDiskANN.insert!(db, "node1")

    # insert node2 (creates adjacency with node1)
    WunDeeDB.LMDiskANN.insert!(db, "node2")

    # if we search near node1's embedding, node1 should come first
    local query1 = Float32[0.11, 0.21, 0.29]
    local res1 = WunDeeDB.LMDiskANN.search(db, query1, topk=2)
    @test "node1" in res1
    @test length(res1) <= 2

    # search near node2's embedding, node2 should come first
    local query2 = Float32[0.77, 0.68, 0.61]
    local res2 = WunDeeDB.LMDiskANN.search(db, query2, topk=2)
    @test "node2" in res2

    #  delete node1
    WunDeeDB.LMDiskANN.delete!(db, "node1")

    #searching near node1's embedding should not yield node1
    local res3 = WunDeeDB.LMDiskANN.search(db, query1, topk=2)
    @test !("node1" in res3)  # node1 is gone
    # node2 might remain

    close_db(db)
    clean_up()
end


@testset "LMDiskANN Empty DB Test" begin
    local TEST_DB = "temp_lmdiskann_empty.sqlite"
    for f in readdir(".")
        if startswith(f, "temp_lmdiskann_empty")
            rm(f; force=true)
        end
    end

    #sets up the DB with lmdiskann but doesn't insert anything
    local init_res = initialize_db(TEST_DB, 3, "Float32"; 
                                   keep_conn_open=true, 
                                   description="Empty DB test",
                                   ann="lmdiskann")
    @test init_res === true

    local db = open_db(TEST_DB, keep_conn_open=true)

    #searching in a empty adjacency should give an empty array in test case
    local query = Float32[0.1, 0.2, 0.3]
    local res = WunDeeDB.LMDiskANN.search(db, query, topk=5)
    @test isempty(res)

    close_db(db)
    rm(TEST_DB; force=true)
end

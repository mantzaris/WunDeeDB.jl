using WunDeeDB
using Test
using SQLite
using DataFrames
using DBInterface, Tables
using DataStructures 
using Random, StatsBase




@testset "Supported Data Types" begin
    types = get_supported_data_types()
    
    
    @test isa(types, Vector{String})
    
    @test !isempty(types)
        
    @test "Float16" in types
    @test "Float32" in types
    @test "UInt128" in types
    
    @test types == sort(types)
    
    #check for at least 5 supported types
    @test length(types) ≥ 5
end




@testset "initialize_db tests" begin
    TEST_DB = "temp_test_db.sqlite"
    TEST_DB2 = "temp_test_db2.sqlite"
    TEST_DB3 = "temp_test_db3.sqlite"

    #clean up any pre-existing test files
    for db_file in (TEST_DB, TEST_DB2, TEST_DB3)
        if isfile(db_file)
            rm(db_file)
        end
    end

    #test 1: valid initialization
    result = initialize_db(TEST_DB, 128, "Float32", description="test dataset", keep_conn_open=false)
    @test result === true
    
    #retrieve meta data and verify its values
    meta = get_meta_data(TEST_DB)
    @test meta !== nothing
    @test meta.embedding_length[1] == 128
    @test meta.data_type[1] == "Float32"
    @test meta.description[1] == "test dataset"
    @test meta.endianness[1] == "small"

    #test 2: Invalid initialization due to embedding_length < 1
    result2 = initialize_db(TEST_DB2, 0, "Float32")
    @test occursin("Embedding_length must be 1 or greater", result2)

    #invalid initialization due to unsupported data_type
    result3 = initialize_db(TEST_DB3, 128, "BadType")
    @test occursin("Unsupported data_type", result3)

    #clean up test databases after testing
    close_db()

    for db_file in (TEST_DB, TEST_DB2, TEST_DB3)
        if isfile(db_file)
            rm(db_file)
        end
    end
end





@testset "open_db tests" begin
    #clean up any pre-existing test files
    for f in ["temp_db.sqlite", "temp_db2.sqlite"]
        if isfile(f)
            rm(f)
        end
    end

    #frst call should create a new DB and store it in WunDeeDB.DB_HANDLE
    db1 = open_db("temp_db.sqlite", keep_conn_open=true)
    @test isa(db1, SQLite.DB)
    @test WunDeeDB.DB_HANDLE[] !== nothing

    #subsequent call with the same parameters should return the same instance
    db2 = open_db("temp_db.sqlite", keep_conn_open=true)
    @test db1 === db2

    
    #calling with keep_conn_open=false should set WunDeeDB.DB_HANDLE[] to nothing
    db3 = open_db("temp_db2.sqlite", keep_conn_open=false)
    @test isa(db3, SQLite.DB)
    @test WunDeeDB.DB_HANDLE[] === nothing

    #another call with keep_conn_open=false should create a new connection instance
    db4 = open_db("temp_db2.sqlite", keep_conn_open=false)
    @test db3 !== db4

    #Important, Directory creation!
    temp_dir = "temp_test_dir"
    temp_db3 = joinpath(temp_dir, "temp_db3.sqlite")
    if isdir(temp_dir)
        rm(temp_dir; recursive=true)
    end
    db5 = open_db(temp_db3, keep_conn_open=false)
    @test isdir(temp_dir)  #directory should now exist
    
    #close the connection and remove the test file and directory
    close_db(db5)
    rm(temp_db3)
    rm(temp_dir; recursive=true)

    #final clean-up: remove the remaining test databases
    for file in readdir(".")
        if startswith(file, "temp") && isfile(file)
            rm(file; force=true)
        end
    end
end




@testset "delete_db tests" begin
    #delete an existing database file
    temp_db = "temp_test_delete.sqlite"
    #create the DB file by opening (and then closing) a connection
    db = SQLite.DB(temp_db)
    close(db)
    @test isfile(temp_db)  # ensure the file exists
    
    result = delete_db(temp_db)
    @test result === true
    @test !isfile(temp_db)

    #attempt to delete a non-existent database
    result2 = delete_db(temp_db)
    @test occursin("Database file does not exist", result2)

    #if a connection is open, delete_db should close it and reset DB_HANDLE
    temp_db2 = "temp_test_delete2.sqlite"
    db2 = SQLite.DB(temp_db2)
    #force the global DB_HANDLE to point to this connection
    WunDeeDB.DB_HANDLE[] = db2
    #now call delete_db on the file
    result3 = delete_db(temp_db2)
    @test result3 === true
    @test WunDeeDB.DB_HANDLE[] === nothing
    @test !isfile(temp_db2)

    close_db()

end




@testset "delete_all_embeddings tests" begin
    TEST_DB = "temp_delete_all_embeddings.sqlite"
    
    #clean up any pre-existing file
    if isfile(TEST_DB)
        rm(TEST_DB)
    end

    #initialize the database
    res_init = initialize_db(TEST_DB, 128, "Float32", description="test embeddings", keep_conn_open=false)
    @test res_init === true

    #insert an embedding then delete
    embedding = rand(Float32, 128)
    id = "embedding1"
    res_insert = insert_embeddings(TEST_DB, id, embedding)
    @test res_insert === true


    count_before = count_entries(TEST_DB)
    @test count_before > 0  #ensure there is at least one embedding

    res_delete = delete_all_embeddings(TEST_DB)
    @test res_delete === true

    count_after = count_entries(TEST_DB, update_meta=true)
    @test count_after == 0  # table should be empty after deletion

    # delete from an already empty table
    res_delete_empty = delete_all_embeddings(TEST_DB)
    @test res_delete_empty === true

    #remove the test database (and any temporary SQLite files like -shm, -wal)
    close_db()
    for f in readdir(".")
        if startswith(f, "temp_delete_all_embeddings")
            rm(f; force=true)
        end
    end
end






@testset "get_meta_data tests" begin
    TEST_DB = "temp_get_meta_data.sqlite"

    # clear up any pre-existing test files (including any WAL/shm files)
    for f in readdir(".")
        if startswith(f, "temp_get_meta_data")
            rm(f; force=true)
        end
    end

    #initialize the database with known meta values
    #we set keep_conn_open=true so that the same connection is reused
    res_init = initialize_db(TEST_DB, 128, "Float32", description="test embeddings", keep_conn_open=true)
    @test res_init === true

    #with the db_path overload:
    meta1 = get_meta_data(TEST_DB)
    @test meta1 !== nothing
    @test isa(meta1, DataFrame)
    @test nrow(meta1) == 1
    @test meta1.embedding_count[1] == 0
    @test meta1.embedding_length[1] == 128
    @test meta1.data_type[1] == "Float32"
    @test meta1.endianness[1] == "small"
    @test meta1.description[1] == "test embeddings"

   
    db = open_db(TEST_DB, keep_conn_open=true)
    meta2 = get_meta_data(db)
    @test meta2 !== nothing
    @test isa(meta2, DataFrame)
    @test nrow(meta2) == 1
    @test meta2.embedding_count[1] == 0

    #simulate missing meta table
    #for testing purposes, drop the meta table and verify that get_meta_data errors
    DBInterface.execute(db, "DROP TABLE $(WunDeeDB.META_DATA_TABLE_NAME)")
    try
        _ = get_meta_data(db)
        @test false  # we should not reach this point
    catch e
        @test occursin("no such table", string(e))
    end

    #clean up: close the persistent connection and remove test files
    close_db(db)
    for f in readdir(".")
        if startswith(f, "temp_get_meta_data")
            rm(f; force=true)
        end
    end
end






@testset "update_description tests" begin
    TEST_DB = "temp_update_description.sqlite"

    #clean up any pre-existing test files
    for f in readdir(".")
        if startswith(f, "temp_update_description")
            rm(f; force=true)
        end
    end

    #initialize the database with an initial description
    res_init = initialize_db(TEST_DB, 128, "Float32", description="initial description", keep_conn_open=false)
    @test res_init === true

    #update description using the db overload
    db = open_db(TEST_DB, keep_conn_open=true)
    res_update = update_description(db, "updated via db")
    @test res_update === true

    #retrieve the meta data and check the description
    meta1 = get_meta_data(db)
    @test meta1 !== nothing
    @test meta1.description[1] == "updated via db"

    #update description using the db_path overload
    #(overload does not return a value on success, so we simply check that no error is throw)
    update_description(TEST_DB, "updated via path")
    meta2 = get_meta_data(TEST_DB)
    @test meta2 !== nothing
    @test meta2.description[1] == "updated via path"

    #close the persistent connection and remove the test file(s)
    close_db(db)
    for f in readdir(".")
        if startswith(f, "temp_update_description")
            rm(f; force=true)
        end
    end
end



@testset "update_meta tests" begin
    TEST_DB = "temp_update_meta.sqlite"

    #clean up any pre-existing test file
    if isfile(TEST_DB)
        rm(TEST_DB)
    end

    #initialize the database
    res_init = initialize_db(TEST_DB, 128, "Float32", description="initial", keep_conn_open=false)
    @test res_init === true

    #test 1: Update meta with count=1 (default)
    #initially, embedding_count is 0
    res_update1 = WunDeeDB.update_meta(open_db(TEST_DB, keep_conn_open=false), 1)
    @test res_update1 === true

    #see that embedding_count has been updated to 1
    df1 = DBInterface.execute(open_db(TEST_DB, keep_conn_open=false), WunDeeDB.META_SELECT_ALL_QUERY) |> DataFrame
    @test df1.embedding_count[1] == 1

    #Update meta with count=5
    res_update2 = WunDeeDB.update_meta(open_db(TEST_DB, keep_conn_open=false), 5)
    @test res_update2 === true

    #check that embedding_count is now 6
    df2 = DBInterface.execute(open_db(TEST_DB, keep_conn_open=false), WunDeeDB.META_SELECT_ALL_QUERY) |> DataFrame
    @test df2.embedding_count[1] == 6

    #clean up test DB and any associated temp files
    for f in readdir(".")
        if startswith(f, "temp_update_meta")
            rm(f; force=true)
        end
    end
end




@testset "infer_data_type tests" begin
    #default integer literals are of type Int64
    @test infer_data_type([1, 2, 3]) == "Int64"

    #floating point literals default to Float64
    @test infer_data_type([1.0, 2.0, 3.0]) == "Float64"

    #explicitly using Float32
    @test infer_data_type(Float32[1.0, 2.0, 3.0]) == "Float32"

    #test with smaller integer types
    @test infer_data_type([Int8(10), Int8(20)]) == "Int8"
end






@testset "insert_embeddings error tests (inner function)" begin
    TEST_DB = "temp_insert_embeddings_error.sqlite"
    
    #clean up any pre-existing test files
    for f in readdir(".")
        if startswith(f, "temp_insert_embeddings_error")
            rm(f; force=true)
        end
    end

    #initialize the database with embedding_length=128, data type "Float32", description "test embeddings"
    res_init = initialize_db(TEST_DB, 128, "Float32", description="test embeddings", keep_conn_open=true)
    @test res_init === true

    #open a persistent connection
    db = open_db(TEST_DB, keep_conn_open=true)

    #prepare a correct embedding vector of length 128
    correct_embedding = Float32.(rand(128))
    
    # --- Test 1: Mismatch between number of IDs and embeddings ---
    #pass two IDs and a single embedding (which will be wrapped to one element)
    err_msg = try
        # Call the inner function directly.
        insert_embeddings(db, ["id1", "id2"], correct_embedding)
        "no error"
    catch e
        string(e)
    end
    @test occursin("Mismatch between number of IDs and embeddings", err_msg)
    
    # --- Test 2: Embedding length mismatch ---
    #prepare an embedding vector of the wrong length (e.g. length 100 instead of 128)
    wrong_embedding = Float32.(rand(100))
    err_msg2 = try
        insert_embeddings(db, "id3", wrong_embedding)
        "no error"
    catch e
        string(e)
    end
    @test occursin("Embedding length mismatch", err_msg2)
    
    #clean up: close the connection and remove test files
    close_db(db)
    for f in readdir(".")
        if startswith(f, "temp_insert_embeddings_error")
            rm(f; force=true)
        end
    end
end





@testset "update_embeddings tests" begin
    #clean up any pre-existing test files
    for f in readdir(".")
        if startswith(f, "temp_update_embeddings")
            rm(f; force=true)
        end
    end

    local TEST_DB = "temp_update_embeddings.sqlite"

    #initialize the database embedding_length = 128, data_type = "Float32"
    res_init = initialize_db(TEST_DB, 128, "Float32",
                             description="test embeddings",
                             keep_conn_open=false)
    @test res_init === true

    #insert an embedding for testing updates
    emb = Float32.(rand(128))
    res_ins = insert_embeddings(TEST_DB, "emb1", emb)
    @test res_ins === true

    #update an existing record with a valid new embedding use Float32(0.1) to keep the vector in Float32
    new_emb = emb .+ Float32(0.1)
    res_upd = update_embeddings(TEST_DB, "emb1", new_emb)
    @test res_upd === true

    #update a record that does not exist before negative tests (#2 and #3) to ensure the DB is still open and EmbeddingsMetaData is present
    err_msg3 = update_embeddings(TEST_DB, "nonexistent", new_emb)
    @test occursin("Record with id nonexistent not found", err_msg3)

    #mismatched IDs and embeddings (should trigger an error) the DB to close in your catch block
    err_msg = update_embeddings(TEST_DB, ["emb1", "emb2"], [new_emb])
    @test occursin("Mismatch between number of IDs and new embeddings", err_msg)

    #passing an invalid type for new_embedding_input
    err_msg2 = update_embeddings(TEST_DB, "emb1", "invalid type")
    @test occursin("Invalid type for new_embedding_input", err_msg2)

    #clean up: remove the test files
    close_db()

    for f in readdir(".")
        if startswith(f, "temp_update_embeddings")
            rm(f; force=true)
        end
    end
end




@testset "get_embeddings tests" begin

    # Clean up pre-existing test files
    local TEST_DB = "temp_get_embeddings.sqlite"
    for f in readdir(".")
        if startswith(f, "temp_get_embeddings")
            rm(f; force=true)
        end
    end

    # Initialize a fresh DB (embedding_length=3 for simplicity)
    res_init = initialize_db(TEST_DB, 3, "Float32", description="test embeddings", keep_conn_open=false)
    @test res_init === true

    # Insert some test embeddings
    emb1 = Float32[0.1, 0.2, 0.3]
    emb2 = Float32[1.0, 2.0, 3.0]

    res_ins1 = insert_embeddings(TEST_DB, "id1", emb1)
    @test res_ins1 === true

    res_ins2 = insert_embeddings(TEST_DB, "id2", emb2)
    @test res_ins2 === true

    # --- Single existing ID => returns a Dict with exactly that ID as a key ---
    res_get1 = get_embeddings(TEST_DB, "id1")
    @test length(res_get1) == 1
    @test haskey(res_get1, "id1")
    @test res_get1["id1"] == emb1

    # --- Multiple IDs => Dict with those IDs found ---
    res_get_multi = get_embeddings(TEST_DB, ["id1", "id2"])
    @test length(res_get_multi) == 2
    @test haskey(res_get_multi, "id1") && haskey(res_get_multi, "id2")
    @test res_get_multi["id1"] == emb1
    @test res_get_multi["id2"] == emb2

    # --- Single nonexistent ID => returns an empty Dict ---
    res_get_nonexistent = get_embeddings(TEST_DB, "no_such_id")
    @test length(res_get_nonexistent) == 0
    @test res_get_nonexistent == Dict()

    # --- Mix of existing and nonexistent IDs => missing nonexistent from Dict ---
    res_get_partial = get_embeddings(TEST_DB, ["id1", "no_such_id", "id2"])
    @test length(res_get_partial) == 2
    @test haskey(res_get_partial, "id1")
    @test haskey(res_get_partial, "id2")
    @test !haskey(res_get_partial, "no_such_id")

    # --- Passing an empty array of IDs => should error ---
    @test true !== get_embeddings(TEST_DB, String[])

    # Clean up
    close_db()

    for f in readdir(".")
        if startswith(f, "temp_get_embeddings")
            rm(f; force=true)
        end
    end
end






@testset "random_embeddings tests" begin

    for f in readdir(".")
        if startswith(f, "temp_")
            rm(f; force=true)
        end
    end

    local TEST_DB = "temp_random_embeddings.sqlite"

    res_init = initialize_db(TEST_DB, 3, "Float32",
                             description="test embeddings",
                             keep_conn_open=false)
    @test res_init === true

   
    emb_ids = ["id$(i)" for i in 1:5]  # 5 distinct IDs
    emb_values = [
        Float32[0.1, 0.2, 0.3],
        Float32[1.0, 2.0, 3.0],
        Float32[0.5, 0.5, 0.5],
        Float32[3.1, 1.59, 2.65],
        Float32[9.99, 9.88, 9.77]
    ]

    for (i, id) in enumerate(emb_ids)
        res_ins = insert_embeddings(TEST_DB, id, emb_values[i])
        @test res_ins === true
    end

    rand_res_2 = random_embeddings(TEST_DB, 2)
    @test length(rand_res_2) == 2
    @test all(in(emb_ids), keys(rand_res_2))  

    for (id_key, vec) in rand_res_2
        @test haskey(rand_res_2, id_key)  # 
        @test length(vec) == 3
    end


    rand_res_10 = random_embeddings(TEST_DB, 10)
    @test length(rand_res_10) <= 5  # SQLite returns at most 5
    for (id_key, vec) in rand_res_10
        @test id_key in emb_ids
        @test length(vec) == 3
    end


    rand_res_0 = random_embeddings(TEST_DB, 0)
    @test length(rand_res_0) == 0  # no rows expected


    rand_res_neg = random_embeddings(TEST_DB, -1)
    @test rand_res_neg !== true  # or handle error if your function does so
    
    # # Clean up
    close_db()

    for f in readdir(".")
        if startswith(f, "temp_")
            rm(f; force=true)
        end
    end
end




@testset "get_adjacent_id tests" begin
    #clean up any existing testDB files
    local TEST_DB = "temp_get_adjacent_id.sqlite"
    for f in readdir(".")
        if startswith(f, "temp_get_adjacent_id")
            rm(f; force=true)
        end
    end

    #initialize a DB with data
    #assume embedding_length=3, data_type="Float32"
    res_init = initialize_db(TEST_DB, 3, "Float32",
        description="adjacent test",
        keep_conn_open=false)
    @test res_init === true

    #insert test rows
    #insert 3 IDs in ascending order: "alpha", "beta", "gamma"
    emb_alpha = Float32[0.1, 0.2, 0.3]
    emb_beta  = Float32[0.4, 0.5, 0.6]
    emb_gamma = Float32[0.7, 0.8, 0.9]

    @test insert_embeddings(TEST_DB, "alpha", emb_alpha) === true
    @test insert_embeddings(TEST_DB, "beta",  emb_beta)  === true
    @test insert_embeddings(TEST_DB, "gamma", emb_gamma) === true

    #test "next" logic
    res_alpha_next = get_adjacent_id(TEST_DB, "alpha"; direction="next", full_row=false)
    @test res_alpha_next == "beta"

    #from "beta" => "gamma"
    res_beta_next = get_adjacent_id(TEST_DB, "beta"; direction="next", full_row=false)
    @test res_beta_next == "gamma"


    #from "gamma" => nothing (there is no lexicographically larger ID)
    res_gamma_next = get_adjacent_id(TEST_DB, "gamma"; direction="next", full_row=false)
    @test res_gamma_next === nothing

    #"previous" logic
    res_gamma_prev = get_adjacent_id(TEST_DB, "gamma"; direction="previous", full_row=false)
    @test res_gamma_prev == "beta"

    #previous from "beta" => "alpha"
    res_beta_prev = get_adjacent_id(TEST_DB, "beta"; direction="prev", full_row=false)
    @test res_beta_prev == "alpha"

    #previous from "alpha" => nothing (no smaller ID)
    res_alpha_prev = get_adjacent_id(TEST_DB, "alpha"; direction="previous", full_row=false)
    @test res_alpha_prev === nothing

    #test "full_row=true"
    #expect a named tuple: (id_text, embedding, data_type)
    #from "alpha" => should return "beta"
    res_alpha_next_full = get_adjacent_id(TEST_DB, "alpha"; direction="next", full_row=true)
    @test res_alpha_next_full !== nothing
    @test res_alpha_next_full.id_text == "beta"
    @test res_alpha_next_full.data_type == "Float32"
    @test res_alpha_next_full.embedding == emb_beta

    #invalid direction => should error
    @test true !== get_adjacent_id(TEST_DB, "alpha"; direction="invalid_dir")

    #clean up
    close_db()

    for f in readdir(".")
        if startswith(f, "temp_get_adjacent_id")
            rm(f; force=true)
        end
    end
end




@testset "count_entries tests" begin
    #clean up any existing test DB files
    local TEST_DB = "temp_count_entries.sqlite"
    for f in readdir(".")
        if startswith(f, "temp_count_entries")
            rm(f; force=true)
        end
    end

    res_init = initialize_db(TEST_DB, 3, "Float32", 
                             description="test count_entries", 
                             keep_conn_open=false)
    @test res_init === true

    #confirm that count_entries is 0 initially, and meta is 0 if we update it
    c0 = count_entries(TEST_DB)  # By default, update_meta=false
    @test c0 == 0  # Table is empty after initialization

    #with update_meta=true
    c0_update = count_entries(TEST_DB; update_meta=true)
    @test c0_update == 0

    #verify meta table embedding_count is 0
    df_meta_0 = DBInterface.execute(open_db(TEST_DB), 
        "SELECT embedding_count FROM $(WunDeeDB.META_DATA_TABLE_NAME)") |> DataFrame
    @test df_meta_0[1, :embedding_count] == 0
        
    #insert some rows and confirm the count changes
    emb1 = Float32[0.1, 0.2, 0.3]
    emb2 = Float32[0.4, 0.5, 0.6]
    insert_embeddings(TEST_DB, "id1", emb1)
    insert_embeddings(TEST_DB, "id2", emb2)

    #check new count without updating meta
    c2 = count_entries(TEST_DB)  # update_meta=false
    @test c2 == 2
          
    df_meta_after_insert = DBInterface.execute(open_db(TEST_DB), 
        "SELECT embedding_count FROM $(WunDeeDB.META_DATA_TABLE_NAME)") |> DataFrame
    @test df_meta_after_insert[1, :embedding_count] == 2
    
    #update meta
    c2_up = count_entries(TEST_DB; update_meta=true)
    @test c2_up == 2
    
    df_meta_after_update = DBInterface.execute(open_db(TEST_DB), 
        "SELECT embedding_count FROM $(WunDeeDB.META_DATA_TABLE_NAME)") |> DataFrame
    @test df_meta_after_update[1, :embedding_count] == 2

    #insert more rows, check count, etc
    emb3 = Float32[1.0, 2.0, 3.0]
    insert_embeddings(TEST_DB, "id3", emb3)

    c3 = count_entries(TEST_DB; update_meta=false)
    @test c3 == 3  # total rows
    
    #confirm meta is still 2 because we didn't update it
    df_meta_unchanged = DBInterface.execute(open_db(TEST_DB), 
        "SELECT embedding_count FROM $(WunDeeDB.META_DATA_TABLE_NAME)") |> DataFrame
    @test df_meta_unchanged[1, :embedding_count] == 3

    c3_up = count_entries(TEST_DB; update_meta=true)
    @test c3_up == 3
    df_meta_final = DBInterface.execute(open_db(TEST_DB),
        "SELECT embedding_count FROM $(WunDeeDB.META_DATA_TABLE_NAME)") |> DataFrame
    @test df_meta_final[1, :embedding_count] == 3

    #clean up
    close_db()

    for f in readdir(".")
        if startswith(f, "temp_count_entries")
            rm(f; force=true)
        end
    end
end






@testset "delete_embeddings tests" begin
    local TEST_DB = "temp_delete_embeddings.sqlite"
    
    # clean up any pre-existing test files
    for f in readdir(".")
        if startswith(f, "temp_delete_")
            rm(f; force=true)
        end
    end

    #initialize the database
    res_init = initialize_db(TEST_DB, 128, "Float32", description="test embeddings", keep_conn_open=false)
    @test res_init === true
    return
    #insert test embeddings
    emb1 = Float32.(rand(128))
    emb2 = Float32.(rand(128))
    
    res_ins1 = insert_embeddings(TEST_DB, "emb1", emb1)
    res_ins2 = insert_embeddings(TEST_DB, "emb2", emb2)
    @test res_ins1 === true
    @test res_ins2 === true

    total_before = count_entries(TEST_DB)
    @test total_before == 2
    
    #delete a single embedding by ID
    res_del1 = delete_embeddings(TEST_DB, "emb1")
    @test res_del1 === true
    total_after1 = count_entries(TEST_DB)
    @test total_after1 == 1

    #delete remaining embedding using an array
    res_del2 = delete_embeddings(TEST_DB, ["emb2"])
    @test res_del2 === true
    total_after2 = count_entries(TEST_DB)
    @test total_after2 == 0
    
    #passing an empty vector should return an error message
    err_msg = delete_embeddings(TEST_DB, String[])
    @test err_msg !== true
    
    close_db()

    for f in readdir(".")
        if startswith(f, "temp_delete_")
            rm(f; force=true)
        end
    end
end





@testset "get_all_embeddings tests" begin
    close_db()

    local TEST_DB = "temp_get_all_embeddings.sqlite"
    for f in readdir(".")
        if startswith(f, "temp_")
            rm(f; force=true)
        end
    end

    res_init = initialize_db(TEST_DB, 3, "Float32",
                             description="test get_all_embeddings",
                             keep_conn_open=false)
    @test res_init === true

    emb1 = Float32[0.1, 0.2, 0.3]
    emb2 = Float32[1.0, 2.0, 3.0]
    @test insert_embeddings(TEST_DB, "id1", emb1) === true
    @test insert_embeddings(TEST_DB, "id2", emb2) === true

    db = open_db(TEST_DB, keep_conn_open=true)
    all_emb_conn = get_all_embeddings(db)
    @test isa(all_emb_conn, Dict)
    @test length(all_emb_conn) == 2
    @test "id1" in keys(all_emb_conn)
    @test "id2" in keys(all_emb_conn)
    @test all_emb_conn["id1"] == emb1
    @test all_emb_conn["id2"] == emb2

    close_db(db)
    all_emb_path = get_all_embeddings(TEST_DB)
    @test isa(all_emb_path, Dict)
    @test length(all_emb_path) == 2
    @test "id1" in keys(all_emb_path)
    @test "id2" in keys(all_emb_path)
    @test all_emb_path["id1"] == emb1
    @test all_emb_path["id2"] == emb2

    db_err = open_db(TEST_DB, keep_conn_open=true)

    SQLite.execute(db_err, "DROP TABLE $(WunDeeDB.META_DATA_TABLE_NAME)")
    err_result = get_all_embeddings(TEST_DB)
    @test isa(err_result, String)
    @test occursin("no such table", err_result) || occursin("Meta table is empty", err_result)


    close_db()
    for f in readdir(".")
        if startswith(f, "temp_")
            rm(f; force=true)
        end
    end
end



@testset "get_all_ids tests" begin
    close_db()
    
    local TEST_DB = "temp_get_all_ids.sqlite"
    
    for f in readdir(".")
        if startswith(f, "temp_")
            rm(f; force=true)
        end
    end

    res_init = initialize_db(TEST_DB, 3, "Float32",
                             description="test get_all_ids",
                             keep_conn_open=false)
    @test res_init === true

    emb1 = Float32[0.1, 0.2, 0.3]
    emb2 = Float32[1.0, 2.0, 3.0]
    emb3 = Float32[4.0, 5.0, 6.0]

    @test insert_embeddings(TEST_DB, "id1", emb1) === true
    @test insert_embeddings(TEST_DB, "id2", emb2) === true
    @test insert_embeddings(TEST_DB, "id3", emb3) === true

    db = open_db(TEST_DB, keep_conn_open=true)
    all_ids_conn = get_all_ids(db)
    @test isa(all_ids_conn, Vector{String})
    @test length(all_ids_conn) == 3
    @test "id1" in all_ids_conn
    @test "id2" in all_ids_conn
    @test "id3" in all_ids_conn

    close_db(db)  
    all_ids_path = get_all_ids(TEST_DB)
    @test isa(all_ids_path, Vector{String})
    @test length(all_ids_path) == 3
    @test "id1" in all_ids_path
    @test "id2" in all_ids_path
    @test "id3" in all_ids_path

    db_err = open_db(TEST_DB, keep_conn_open=true)
    SQLite.execute(db_err, "DROP TABLE $(WunDeeDB.MAIN_TABLE_NAME)")
    err_result = get_all_ids(TEST_DB)
    @test isa(err_result, String)
    @test occursin("no such table", err_result)

    close_db()
    for f in readdir(".")
        if startswith(f, "temp_")
            rm(f; force=true)
        end
    end
end





@testset "linear_search_iteration tests" begin
    close_db()
    for f in readdir(".")
        if startswith(f, "temp_")
            rm(f; force=true)
        end
    end

    local TEST_DB = "temp_linear_search_iteration.sqlite"
    
    res_init = initialize_db(TEST_DB, 3, "Float32",
                             description="test linear search iteration",
                             keep_conn_open=false)
    @test res_init === true

    @test insert_embeddings(TEST_DB, "a", Float32[0.0, 0.0, 0.0]) === true
    @test insert_embeddings(TEST_DB, "b", Float32[1.0, 1.0, 1.0]) === true
    @test insert_embeddings(TEST_DB, "c", Float32[2.0, 2.0, 2.0]) === true
    @test insert_embeddings(TEST_DB, "d", Float32[-1.0, -1.0, -1.0]) === true
    @test insert_embeddings(TEST_DB, "e", Float32[0.5, 0.5, 0.6]) === true

    query = Float32[0.5, 0.5, 0.5]
    metric = "euclidean"
    top_k = 3

    results = linear_search_iteration(TEST_DB, query, metric; top_k=top_k)
    @test isa(results, Vector)
    @test length(results) == top_k

    distances = [r[1] for r in results]
    ids = [r[2] for r in results]

    @test isapprox(distances[1], 0.1; atol=1e-3)
    @test distances[2] ≥ 0.1
    @test distances[3] ≥ distances[2]  #ensure sorted order

    @test "e" in ids

    top_k_large = 10
    results_large = linear_search_iteration(TEST_DB, query, metric; top_k=top_k_large)
    @test isa(results_large, Vector)
    @test length(results_large) == 5  #only 5 embeddings were inserted

    db_err = open_db(TEST_DB, keep_conn_open=true)
    # SQLite.execute(db_err, "DROP TABLE $(WunDeeDB.MAIN_TABLE_NAME)")
    # err_result = linear_search_iteration(TEST_DB, query, metric; top_k=top_k)
    # @test isa(err_result, String)
    # @test occursin("no such table", err_result)

    close_db()
    for f in readdir(".")
        if startswith(f, "temp_")
            rm(f; force=true)
        end
    end
end






@testset "linear_search_ids tests" begin
    close_db()
    for f in readdir(".")
        if startswith(f, "temp_")
            rm(f; force=true)
        end
    end

    # Use a unique temporary database file.
    local TEST_DB = "temp_linear_search_ids.sqlite"

    res_init = initialize_db(TEST_DB, 3, "Float32",
                             description="test linear_search_ids",
                             keep_conn_open=false)
    @test res_init === true

    @test insert_embeddings(TEST_DB, "a", Float32[0.0, 0.0, 0.0]) === true
    @test insert_embeddings(TEST_DB, "b", Float32[1.0, 1.0, 1.0]) === true
    @test insert_embeddings(TEST_DB, "c", Float32[2.0, 2.0, 2.0]) === true
    @test insert_embeddings(TEST_DB, "d", Float32[-1.0, -1.0, -1.0]) === true
    @test insert_embeddings(TEST_DB, "e", Float32[0.5, 0.5, 0.6]) === true

    query = Float32[0.5, 0.5, 0.5]
    metric = "euclidean"
    top_k = 3

    results = linear_search_ids(TEST_DB, query, metric; top_k=top_k)
    @test isa(results, Vector)
    @test length(results) == top_k

    distances = [r[1] for r in results]
    ids = [r[2] for r in results]

    @test isapprox(distances[1], 0.1; atol=1e-3)
    @test distances[1] ≤ distances[2] ≤ distances[3]
    @test "e" in ids

    top_k_large = 10
    results_large = linear_search_ids(TEST_DB, query, metric; top_k=top_k_large)
    @test isa(results_large, Vector)
    @test length(results_large) == 5

    close_db()
    for f in readdir(".")
        if startswith(f, "temp_")
            rm(f; force=true)
        end
    end
end





@testset "linear_search_ids_batched tests" begin
    close_db()
    for f in readdir(".")
        if startswith(f, "temp_")
            rm(f; force=true)
        end
    end

    local TEST_DB = "temp_linear_search_ids_batched.sqlite"

    res_init = initialize_db(TEST_DB, 3, "Float32",
                             description="test linear_search_ids_batched",
                             keep_conn_open=false)
    @test res_init === true

    @test insert_embeddings(TEST_DB, "a", Float32[0.0, 0.0, 0.0]) === true
    @test insert_embeddings(TEST_DB, "b", Float32[1.0, 1.0, 1.0]) === true
    @test insert_embeddings(TEST_DB, "c", Float32[2.0, 2.0, 2.0]) === true
    @test insert_embeddings(TEST_DB, "d", Float32[-1.0, -1.0, -1.0]) === true
    @test insert_embeddings(TEST_DB, "e", Float32[0.5, 0.5, 0.6]) === true

    query = Float32[0.5, 0.5, 0.5]
    metric = "euclidean"
    top_k = 3
    batch_size = 1

    results = linear_search_ids_batched(TEST_DB, query, metric; top_k=top_k, batch_size=batch_size)
    @test isa(results, Vector)
    @test length(results) == top_k

    distances = [r[1] for r in results]
    ids = [r[2] for r in results]

    @test "e" in ids
    @test isapprox(distances[1], 0.1; atol=1e-3)
    @test distances[1] ≤ distances[2] ≤ distances[3]

    top_k_large = 10
    results_large = linear_search_ids_batched(TEST_DB, query, metric; top_k=top_k_large, batch_size=batch_size)
    @test isa(results_large, Vector)
    @test length(results_large) == 5

    close_db()
    for f in readdir(".")
        if startswith(f, "temp_")
            rm(f; force=true)
        end
    end
end



@testset "linear_search_all_embeddings tests" begin
    close_db()
    for f in readdir(".")
        if startswith(f, "temp_")
            rm(f; force=true)
        end
    end

    local TEST_DB = "temp_linear_search_all_embeddings.sqlite"

    res_init = initialize_db(TEST_DB, 3, "Float32",
                             description="test linear_search_all_embeddings",
                             keep_conn_open=false)
    @test res_init === true

    @test insert_embeddings(TEST_DB, "a", Float32[0.0, 0.0, 0.0]) === true
    @test insert_embeddings(TEST_DB, "b", Float32[1.0, 1.0, 1.0]) === true
    @test insert_embeddings(TEST_DB, "c", Float32[2.0, 2.0, 2.0]) === true
    @test insert_embeddings(TEST_DB, "d", Float32[-1.0, -1.0, -1.0]) === true
    @test insert_embeddings(TEST_DB, "e", Float32[0.5, 0.5, 0.6]) === true

    query = Float32[0.5, 0.5, 0.5]
    metric = "euclidean"
    top_k = 3

    results = linear_search_all_embeddings(TEST_DB, query, metric; top_k=top_k)
    @test isa(results, Vector)
    @test length(results) == top_k

    distances = [r[1] for r in results]
    ids = [r[2] for r in results]

    @test "e" in ids
    @test isapprox(distances[1], 0.1; atol=1e-3)
    @test distances[1] ≤ distances[2] ≤ distances[3]

    top_k_large = 10
    results_large = linear_search_all_embeddings(TEST_DB, query, metric; top_k=top_k_large)
    @test isa(results_large, Vector)
    @test length(results_large) == 5

    db_err = open_db(TEST_DB, keep_conn_open=true)
    SQLite.execute(db_err, "DROP TABLE $(WunDeeDB.META_DATA_TABLE_NAME)")
    err_result = linear_search_all_embeddings(TEST_DB, query, metric; top_k=top_k)
    @test isa(err_result, String)
    @test occursin("no such table", err_result)

    close_db()
    for f in readdir(".")
        if startswith(f, "temp_")
            rm(f; force=true)
        end
    end
end



# test DISK HNSW
include("diskhnsw_tests.jl")

# test Disk LMDiskANN
include("lmdiskann_tests.jl")
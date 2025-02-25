using WunDeeDB
using Test
using SQLite
using DataFrames
using DBInterface, Tables




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
    println(res_insert)
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





@testset "delete_embeddings tests" begin
    local TEST_DB = "temp_delete_embeddings.sqlite"
    
    # clean up any pre-existing test files
    for f in readdir(".")
        if startswith(f, "temp_delete_embeddings")
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
    
    println("foo")
    #passing an empty vector should return an error message
    err_msg = delete_embeddings(TEST_DB, String[])
    @test err_msg !== true
    
    for f in readdir(".")
        if startswith(f, "temp_delete_embeddings")
            rm(f; force=true)
        end
    end
end






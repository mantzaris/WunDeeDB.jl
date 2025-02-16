using WunDeeDB
using Test
using SQLite

using DBInterface, Tables

@testset "open_db Tests" begin
    #make a temporary directory and a db file path.
    tempdir = mktempdir()

    #see that a new directory is created if it doesn't exist
    non_existent_dir = joinpath(tempdir, "nonexistent_subdir")
    db_path1 = joinpath(non_existent_dir, "testdb1.sqlite")
    @test !isdir(non_existent_dir)#shouldn't exist here now
    db1 = WunDeeDB.open_db(db_path1)
    @test isdir(non_existent_dir) #ut should now be here created
    @test isa(db1, SQLite.DB)

    #check that the PRAGMA settings are correctly applied.
    q = SQLite.execute(db1, "PRAGMA journal_mode;")
    journal_mode = collect(q)[1][1]
    #can be either "wal" or "100" depending on the environment.
    @test lowercase(string(journal_mode)) in ["wal", "100"]

    q2 = SQLite.execute(db1, "PRAGMA synchronous;")
    synchronous_val = collect(q2)[1][1]
    #either 1 or 100.
    @test synchronous_val in (1, 100)

    WunDeeDB.close_db(db1)

    #test that a file was created.
    @test isfile(db_path1)

    #test that open_db returns a valid connection for an existing file.
    db2 = WunDeeDB.open_db(db_path1)
    @test isa(db2, SQLite.DB)
    WunDeeDB.close_db(db2)

    #cleanup the temporary directory!!
    rm(tempdir; force=true, recursive=true)
end




@testset "initialize_db Basic Table Creation Tests" begin
    #make a temporary directory for the database file
    tempdir = mktempdir()
    try
        #set the database file path within the temporary directory
        db_path = joinpath(tempdir, "testdb.sqlite")
        collection_name = "test_collection"

        result = WunDeeDB.initialize_db(db_path, collection_name)
        @test result == "true"

        @test isfile(db_path)

        db = SQLite.DB(db_path)

        #verify that the main table exists
        main_table_query = SQLite.execute(db, """
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' AND name='$collection_name';
        """)
        main_tables = collect(main_table_query)
        @test length(main_tables) == 1

        #verify that the meta table exists
        meta_table_name = "$(collection_name)_meta"
        meta_table_query = SQLite.execute(db, """
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' AND name='$meta_table_name';
        """)
        meta_tables = collect(meta_table_query)
        @test length(meta_tables) == 1

        SQLite.close(db)
    finally
        #cleanup the temporary directory
        rm(tempdir; force=true, recursive=true)
    end
end




@testset "WunDeeDB.update_meta Tests" begin
    db = SQLite.DB()

    #meta table name and the embedding length.
    meta_table = "test_meta"
    embedding_length = 128

    #create the meta table following the expected schema
    meta_stmt = """
    CREATE TABLE IF NOT EXISTS $(meta_table) (
         row_num BIGINT,
         vector_length INT
    )
    """
    SQLite.execute(db, meta_stmt)

    #test 1: an empty table, update_meta should insert a row with row_num = 1
    WunDeeDB.update_meta(db, meta_table, embedding_length)
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test length(rows) == 1
    @test rows[1].row_num == 1
    @test rows[1].vector_length == embedding_length

    #test 2: update_meta again should update the row_num (increment it)
    WunDeeDB.update_meta(db, meta_table, embedding_length)
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test length(rows) == 1
    @test rows[1].row_num == 2
    @test rows[1].vector_length == embedding_length

    #test 3: calling update_meta with a different embedding_length should throw an error
    err = nothing
    try
       WunDeeDB.update_meta(db, meta_table, embedding_length + 1)
    catch e
       err = e
    end
    @test err !== nothing
    @test occursin("Vector length mismatch", string(err))

    SQLite.close(db)
end





@testset "WunDeeDB.update_meta_bulk Tests" begin
    db = SQLite.DB()

    #define the meta table name and the embedding length
    meta_table = "test_meta_bulk"
    embedding_length = 128

    #create the meta table following the expected schema
    meta_stmt = """
    CREATE TABLE IF NOT EXISTS $(meta_table) (
         row_num BIGINT,
         vector_length INT
    )
    """
    SQLite.execute(db, meta_stmt)

    #test 1: With an empty table, update_meta_bulk should insert a row with row_num = count
    count1 = 5
    WunDeeDB.update_meta_bulk(db, meta_table, embedding_length, count1)
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test length(rows) == 1
    @test rows[1].row_num == count1
    @test rows[1].vector_length == embedding_length

    #test 2 Calling update_meta_bulk again should update the row_num by adding count
    count2 = 3
    WunDeeDB.update_meta_bulk(db, meta_table, embedding_length, count2)
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test length(rows) == 1
    @test rows[1].row_num == count1 + count2
    @test rows[1].vector_length == embedding_length

    #test 3 Calling update_meta_bulk with a different embedding_length should throw an error
    err = nothing
    try
       WunDeeDB.update_meta_bulk(db, meta_table, embedding_length + 1, count1)
    catch e
       err = e
    end
    @test err !== nothing
    @test occursin("Vector length mismatch", string(err))

    SQLite.close(db)
end




@testset "WunDeeDB.update_meta_delete Tests" begin
    db = SQLite.DB()

    #define the meta table name and a sample embedding length
    meta_table = "test_meta_delete"
    embedding_length = 128

    #create the meta table following the expected schema
    meta_stmt = """
    CREATE TABLE IF NOT EXISTS $(meta_table) (
         row_num BIGINT,
         vector_length INT
    )
    """
    SQLite.execute(db, meta_stmt)

    #pre-insert a row with row_num = 3 and vector_length = embedding_length
    insert_stmt = """
    INSERT INTO $(meta_table) (row_num, vector_length)
    VALUES (?, ?)
    """
    SQLite.execute(db, insert_stmt, (3, embedding_length))

    #test 1 call update_meta_delete once; row_num should decrement by 1
    WunDeeDB.update_meta_delete(db, meta_table)
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test length(rows) == 1
    @test rows[1].row_num == 2
    @test rows[1].vector_length == embedding_length

    #test 2 call update_meta_delete repeatedly until row_num becomes 0
    WunDeeDB.update_meta_delete(db, meta_table)  #row_num becomes 1
    WunDeeDB.update_meta_delete(db, meta_table)  #row_num becomes 0 and vector_length resets to NULL
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test length(rows) == 1
    @test rows[1].row_num == 0
    #SQLite.jl converts SQL NULL to missing; check
    @test rows[1].vector_length === missing || isnothing(rows[1].vector_length)

    #test 3 if the meta table empty, update_meta_delete should do nothing and not error
    SQLite.execute(db, "DELETE FROM $(meta_table)")
    WunDeeDB.update_meta_delete(db, meta_table)

    SQLite.close(db)
end



#########################################

# --- Test Set for insert_embedding using a DB handle ---
@testset "WunDeeDB.insert_embedding Tests (db handle)" begin
    # Create an in-memory SQLite database.
    db = SQLite.DB()

    # Define the collection and meta table names.
    collection_name = "test_embeddings"
    meta_table = "$(collection_name)_meta"

    # Create the collection table.
    stmt_coll = """
    CREATE TABLE IF NOT EXISTS $(collection_name) (
         id_text TEXT,
         embedding_json TEXT,
         data_type TEXT
    )
    """
    SQLite.execute(db, stmt_coll)

    # Create the meta table.
    stmt_meta = """
    CREATE TABLE IF NOT EXISTS $(meta_table) (
         row_num BIGINT,
         vector_length INT
    )
    """
    SQLite.execute(db, stmt_meta)

    # Test inserting an embedding with no explicit data_type.
    # (Assume infer_data_type returns "Float64" for a Float64 vector.)
    id_text = "test1"
    embedding = [1.0, 2.0, 3.0]  # length = 3
    msg = WunDeeDB.insert_embedding(db, collection_name, id_text, embedding)
    @test msg == "true"

    # Verify the record in the collection table.
    coll_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(collection_name)")))
    @test length(coll_rows) == 1
    @test coll_rows[1].id_text == id_text
    # Check that the JSON contains the embedding values (a simple substring check).
    @test occursin("1.0", coll_rows[1].embedding_json)
    @test coll_rows[1].data_type == "Float64"

    # Verify the meta table: after one insertion, row_num should be 1 and vector_length should equal the embedding length.
    meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test length(meta_rows) == 1
    @test meta_rows[1].row_num == 1      # 1 insertion so far
    @test meta_rows[1].vector_length == 3

    SQLite.close(db)
end

# --- Test Set for insert_embedding using a db_path ---
@testset "WunDeeDB.insert_embedding Tests (db_path overload)" begin
    # Create a temporary directory and database file.
    temp_dir = mktempdir()
    db_path = joinpath(temp_dir, "test.sqlite")
    collection_name = "test_embeddings2"
    meta_table = "$(collection_name)_meta"

    # Open a database and create the necessary tables.
    db = SQLite.DB(db_path)
    stmt_coll = "CREATE TABLE IF NOT EXISTS $(collection_name) (id_text TEXT, embedding_json TEXT, data_type TEXT)"
    SQLite.execute(db, stmt_coll)
    stmt_meta = """
    CREATE TABLE IF NOT EXISTS $(meta_table) (
         row_num BIGINT,
         vector_length INT
    )
    """
    SQLite.execute(db, stmt_meta)
    SQLite.close(db)

    # Test inserting an embedding via the db_path version.
    id_text = "test2"
    embedding = [4, 5, 6, 7]  # length = 4
    msg = WunDeeDB.insert_embedding(db_path, collection_name, id_text, embedding)
    @test msg == "true"

    # Open the db again to verify.
    db = SQLite.DB(db_path)
    coll_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(collection_name)")))
    @test length(coll_rows) == 1
    @test coll_rows[1].id_text == id_text
    @test occursin("4", coll_rows[1].embedding_json)

    meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test length(meta_rows) == 1
    @test meta_rows[1].row_num == 1       # only one insertion has been made
    @test meta_rows[1].vector_length == 4

    SQLite.close(db)
    rm(temp_dir; force=true, recursive=true)
end

# --- Test Set for explicit data_type ---
@testset "WunDeeDB.insert_embedding Tests (explicit data_type)" begin
    db = SQLite.DB()
    collection_name = "test_embeddings3"
    meta_table = "$(collection_name)_meta"

    # Create the tables.
    SQLite.execute(db, "CREATE TABLE IF NOT EXISTS $(collection_name) (id_text TEXT, embedding_json TEXT, data_type TEXT)")
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS $(meta_table) (
             row_num BIGINT,
             vector_length INT
        )
    """)

    id_text = "test3"
    embedding = [7.0, 8.0]  # length = 2
    data_type = "CustomType"
    msg = WunDeeDB.insert_embedding(db, collection_name, id_text, embedding; data_type=data_type)
    @test msg == "true"

    coll_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(collection_name)")))
    @test length(coll_rows) == 1
    @test coll_rows[1].data_type == data_type

    meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test length(meta_rows) == 1
    @test meta_rows[1].row_num == 1       # 1 insertion only
    @test meta_rows[1].vector_length == 2

    SQLite.close(db)
end

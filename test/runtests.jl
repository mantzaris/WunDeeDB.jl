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
#using a DB handle
@testset "WunDeeDB.insert_embedding Tests (db handle)" begin
    db = SQLite.DB()

    collection_name = "test_embeddings"
    meta_table = "$(collection_name)_meta"

    stmt_coll = """
    CREATE TABLE IF NOT EXISTS $(collection_name) (
         id_text TEXT,
         embedding_json TEXT,
         data_type TEXT
    )
    """
    SQLite.execute(db, stmt_coll)

    stmt_meta = """
    CREATE TABLE IF NOT EXISTS $(meta_table) (
         row_num BIGINT,
         vector_length INT
    )
    """
    SQLite.execute(db, stmt_meta)

    id_text = "test1"
    embedding = [1.0, 2.0, 3.0] 
    msg = WunDeeDB.insert_embedding(db, collection_name, id_text, embedding)
    @test msg == "true"

    coll_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(collection_name)")))
    @test length(coll_rows) == 1
    @test coll_rows[1].id_text == id_text

    @test occursin("1.0", coll_rows[1].embedding_json)
    @test coll_rows[1].data_type == "Float64"


    meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test length(meta_rows) == 1
    @test meta_rows[1].row_num == 1      # 1 insertion so far
    @test meta_rows[1].vector_length == 3

    SQLite.close(db)
end

#using a db_path ---
@testset "WunDeeDB.insert_embedding Tests (db_path overload)" begin
    temp_dir = mktempdir()
    db_path = joinpath(temp_dir, "test.sqlite")
    collection_name = "test_embeddings2"
    meta_table = "$(collection_name)_meta"

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

    id_text = "test2"
    embedding = [4, 5, 6, 7]  # length = 4
    msg = WunDeeDB.insert_embedding(db_path, collection_name, id_text, embedding)
    @test msg == "true"

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


@testset "WunDeeDB.insert_embedding Tests (explicit data_type)" begin
    db = SQLite.DB()
    collection_name = "test_embeddings3"
    meta_table = "$(collection_name)_meta"

    SQLite.execute(db, "CREATE TABLE IF NOT EXISTS $(collection_name) (id_text TEXT, embedding_json TEXT, data_type TEXT)")
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS $(meta_table) (
             row_num BIGINT,
             vector_length INT
        )
    """)

    id_text = "test3"
    embedding = [7.0, 8.0]
    data_type = "CustomType"
    msg = WunDeeDB.insert_embedding(db, collection_name, id_text, embedding; data_type=data_type)
    @test msg == "true"

    coll_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(collection_name)")))
    @test length(coll_rows) == 1
    @test coll_rows[1].data_type == data_type

    meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test length(meta_rows) == 1
    @test meta_rows[1].row_num == 1
    @test meta_rows[1].vector_length == 2

    SQLite.close(db)
end


####################
####################

@testset "WunDeeDB.bulk_insert_embedding Tests (DB handle)" begin

    db = SQLite.DB()


    collection_name = "test_bulk_embeddings"
    meta_table = "$(collection_name)_meta"

    # make the collection table.
    stmt_coll = """
    CREATE TABLE IF NOT EXISTS $(collection_name) (
         id_text TEXT,
         embedding_json TEXT,
         data_type TEXT
    )
    """
    SQLite.execute(db, stmt_coll)

    # make the meta table.
    stmt_meta = """
    CREATE TABLE IF NOT EXISTS $(meta_table) (
         row_num BIGINT,
         vector_length INT
    )
    """
    SQLite.execute(db, stmt_meta)

    # make test data
    id_texts = ["id1", "id2", "id3"]
    embeddings = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]
    #assume infer_data_type returns "Float64" for a Float64 vector
    
    # see successful bulk insert
    msg = WunDeeDB.bulk_insert_embedding(db, collection_name, id_texts, embeddings)
    @test msg == "true"

    #check the collection table
    coll_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(collection_name)")))
    @test length(coll_rows) == 3
    @test all(x -> x.data_type == "Float64", coll_rows)

    #check the meta table update
    meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test length(meta_rows) == 1
    #meta table's row_num to be equal to the number of inserted embeddings (3),
    # and the vector_length should equal the length of the first embedding (3).
    @test meta_rows[1].row_num == 3
    @test meta_rows[1].vector_length == 3

    #test error: Mismatch between number of IDs and embeddings!
    id_texts_bad = ["id1", "id2"]
    @test_throws ErrorException WunDeeDB.bulk_insert_embedding(db, collection_name, id_texts_bad, embeddings)

    #TEST ERROR for Inconsistent embedding lengths!
    embeddings_bad = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0],
        [7.0, 8.0, 9.0]
    ]
    @test_throws ErrorException WunDeeDB.bulk_insert_embedding(db, collection_name, id_texts, embeddings_bad)

    #test error: Exceeding BULK_LIMIT (simulate by using a low limit)
    #for testing, we assume BULK_LIMIT is defined; if needed, set a temporary limit
    @test_throws ErrorException WunDeeDB.bulk_insert_embedding(db, collection_name, ["a", "b", "c"], [[1],[2],[3]])

    SQLite.close(db)
end

@testset "WunDeeDB.bulk_insert_embedding Tests (db_path overload)" begin
    #create a temporary directory and file-based database
    temp_dir = mktempdir()
    db_path = joinpath(temp_dir, "test_bulk.sqlite")
    collection_name = "test_bulk_embeddings2"
    meta_table = "$(collection_name)_meta"

    #open a database and create the necessary tables
    db = SQLite.DB(db_path)
    SQLite.execute(db, "CREATE TABLE IF NOT EXISTS $(collection_name) (id_text TEXT, embedding_json TEXT, data_type TEXT)")
    SQLite.execute(db, """
    CREATE TABLE IF NOT EXISTS $(meta_table) (
         row_num BIGINT,
         vector_length INT
    )
    """)
    SQLite.close(db)

    #test successful bulk insert via the db_path overload
    id_texts = ["alpha", "beta"]
    embeddings = [
        [10, 20, 30],
        [40, 50, 60]
    ]
    msg = WunDeeDB.bulk_insert_embedding(db_path, collection_name, id_texts, embeddings)
    @test msg == "true"

    db = SQLite.DB(db_path)
    coll_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(collection_name)")))
    @test length(coll_rows) == 2

    meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test length(meta_rows) == 1
    @test meta_rows[1].row_num == 2
    @test meta_rows[1].vector_length == 3

    SQLite.close(db)
    rm(temp_dir; force=true, recursive=true)
end


#######################
#######################


@testset "WunDeeDB.delete_embedding Tests (DB handle)" begin
    db = SQLite.DB()
    
    
    collection_name = "test_embeddings_delete"
    meta_table = "$(collection_name)_meta"
    
    
    stmt_coll = """
    CREATE TABLE IF NOT EXISTS $(collection_name) (
         id_text TEXT,
         embedding_json TEXT,
         data_type TEXT
    )
    """
    SQLite.execute(db, stmt_coll)
    
    stmt_meta = """
    CREATE TABLE IF NOT EXISTS $(meta_table) (
         row_num BIGINT,
         vector_length INT
    )
    """
    SQLite.execute(db, stmt_meta)
    
    #pre-insert a record into the collectiontable
    insert_stmt = """
    INSERT INTO $(collection_name) (id_text, embedding_json, data_type)
    VALUES (?, ?, ?)
    """
    SQLite.execute(db, insert_stmt, ("record1", "{}", "Float64"))
    
    #pre-insert a meta record
    meta_insert = """
    INSERT INTO $(meta_table) (row_num, vector_length)
    VALUES (?, ?)
    """
    SQLite.execute(db, meta_insert, (1, 3))
    
    #test 1: Successful deletion
    msg = WunDeeDB.delete_embedding(db, collection_name, "record1")
    @test msg == "true"
    
    #check that the record was deleted
    coll_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(collection_name)")))
    @test isempty(coll_rows)
    
    #check that the meta table was updated
    meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test length(meta_rows) == 1
    @test meta_rows[1].row_num == 0
    @test meta_rows[1].vector_length === missing || isnothing(meta_rows[1].vector_length)
    
    #test 2: attempt to delete a non-existent record should throw an error
    @test_throws ErrorException WunDeeDB.delete_embedding(db, collection_name, "nonexistent")
    
    SQLite.close(db)
end

@testset "WunDeeDB.delete_embedding Tests (db_path overload)" begin
    #create a temporary directory and file-based database
    temp_dir = mktempdir()
    db_path = joinpath(temp_dir, "test_delete.sqlite")
    collection_name = "test_embeddings_delete2"
    meta_table = "$(collection_name)_meta"
    
    #open a database and create the necessary tables
    db = SQLite.DB(db_path)
    SQLite.execute(db, "CREATE TABLE IF NOT EXISTS $(collection_name) (id_text TEXT, embedding_json TEXT, data_type TEXT)")
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS $(meta_table) (
             row_num BIGINT,
             vector_length INT
        )
    """)
    
    #insert a record and a meta record
    SQLite.execute(db, "INSERT INTO $(collection_name) (id_text, embedding_json, data_type) VALUES (?, ?, ?)", ("record2", "{}", "Float64"))
    SQLite.execute(db, "INSERT INTO $(meta_table) (row_num, vector_length) VALUES (?, ?)", (1, 3))
    SQLite.close(db)
    
    #test deletion using the db_path overload
    msg = WunDeeDB.delete_embedding(db_path, collection_name, "record2")
    @test msg == "true"
    
    #check results
    db = SQLite.DB(db_path)
    coll_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(collection_name)")))
    @test isempty(coll_rows)
    
    meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test length(meta_rows) == 1
    @test meta_rows[1].row_num == 0
    @test meta_rows[1].vector_length === missing || isnothing(meta_rows[1].vector_length)
    
    SQLite.close(db)
    rm(temp_dir; force=true, recursive=true)
end


################
################


@testset "WunDeeDB.bulk_delete_embedding Tests (DB handle)" begin
    
    db = SQLite.DB()
    
    
    collection_name = "test_bulk_delete"
    meta_table = "$(collection_name)_meta"
    
    
    stmt_coll = """
    CREATE TABLE IF NOT EXISTS $(collection_name) (
         id_text TEXT,
         embedding_json TEXT,
         data_type TEXT
    )
    """
    SQLite.execute(db, stmt_coll)
    
    
    stmt_meta = """
    CREATE TABLE IF NOT EXISTS $(meta_table) (
         row_num BIGINT,
         vector_length INT
    )
    """
    SQLite.execute(db, stmt_meta)
    
    #pre-insert 5 records into the collection table
    insert_stmt = """
    INSERT INTO $(collection_name) (id_text, embedding_json, data_type)
    VALUES (?, ?, ?)
    """
    ids = ["a", "b", "c", "d", "e"]
    for id in ids
        SQLite.execute(db, insert_stmt, (id, "{}", "Float64"))
    end
    
    #pre-insert a meta record with row_num = 5 and vector_length = 3
    meta_insert = """
    INSERT INTO $(meta_table) (row_num, vector_length)
    VALUES (?, ?)
    """
    SQLite.execute(db, meta_insert, (5, 3))
    
    #test 1: Bulk delete two records
    msg = WunDeeDB.bulk_delete_embedding(db, collection_name, ["b", "d"])
    @test msg == "true"
    
    #check that the specified rows were deleted
    coll_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(collection_name)")))
    remaining_ids = sort([row.id_text for row in coll_rows])
    @test remaining_ids == sort(["a", "c", "e"])
    
    # checkl that the meta table was updated
    #since 2 records were deleted from an initial count of 5
    # the new row_num should be 3
    meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test length(meta_rows) == 1
    @test meta_rows[1].row_num == 3
    @test meta_rows[1].vector_length == 3
    BULK_LIMIT=1000
    #test 2: Exceeding BULK_LIMIT should throw an error
    #simulate by passing more IDs than BULK_LIMIT

    id_texts_over = [ "x$(i)" for i in 1:(BULK_LIMIT+1) ]
    @test_throws ErrorException WunDeeDB.bulk_delete_embedding(db, collection_name, id_texts_over)
    
    SQLite.close(db)
end

@testset "WunDeeDB.bulk_delete_embedding Tests (db_path overload)" begin
    #create a temporary directory and file-based database
    temp_dir = mktempdir()
    db_path = joinpath(temp_dir, "test_bulk_delete.sqlite")
    collection_name = "test_bulk_delete2"
    meta_table = "$(collection_name)_meta"
    
    #open a database and create the necessary tables
    db = SQLite.DB(db_path)
    SQLite.execute(db, "CREATE TABLE IF NOT EXISTS $(collection_name) (id_text TEXT, embedding_json TEXT, data_type TEXT)")
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS $(meta_table) (
             row_num BIGINT,
             vector_length INT
        )
    """)
    
    #insert 3 records and a meta record
    for id in ["p", "q", "r"]
        SQLite.execute(db, "INSERT INTO $(collection_name) (id_text, embedding_json, data_type) VALUES (?, ?, ?)", (id, "{}", "Float64"))
    end
    SQLite.execute(db, "INSERT INTO $(meta_table) (row_num, vector_length) VALUES (?, ?)", (3, 3))
    SQLite.close(db)
    
    #db_path overload to delete two records
    msg = WunDeeDB.bulk_delete_embedding(db_path, collection_name, ["p", "r"])
    @test msg == "true"
    
    #reopen the database to check
    db = SQLite.DB(db_path)
    coll_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(collection_name)")))
    @test length(coll_rows) == 1
    @test coll_rows[1].id_text == "q"
    
    meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test length(meta_rows) == 1
    #deleted 2 out of 3, the new meta row_num should be 1
    @test meta_rows[1].row_num == 1
    @test meta_rows[1].vector_length == 3
    
    SQLite.close(db)
    rm(temp_dir; force=true, recursive=true)
end



#################################
#################################



@testset "WunDeeDB.update_embedding Tests (DB handle)" begin

    db = SQLite.DB()


    collection_name = "test_update_embedding"
    meta_table = "$(collection_name)_meta"


    stmt_coll = """
    CREATE TABLE IF NOT EXISTS $(collection_name) (
         id_text TEXT,
         embedding_json TEXT,
         data_type TEXT
    )
    """
    SQLite.execute(db, stmt_coll)


    stmt_meta = """
    CREATE TABLE IF NOT EXISTS $(meta_table) (
         row_num BIGINT,
         vector_length INT
    )
    """
    SQLite.execute(db, stmt_meta)


    insert_stmt = """
    INSERT INTO $(collection_name) (id_text, embedding_json, data_type)
    VALUES (?, ?, ?)
    """
    SQLite.execute(db, insert_stmt, ("record1", "[1,2,3]", "Float64"))
    
    #insert a meta record indicating that the embedding dimension is 3
    meta_insert = """
    INSERT INTO $(meta_table) (row_num, vector_length)
    VALUES (?, ?)
    """
    SQLite.execute(db, meta_insert, (1, 3))

    #Test 1: Successful update
    new_embedding = [4.0, 5.0, 6.0]  # Length 3 (matches meta).
    msg = WunDeeDB.update_embedding(db, collection_name, "record1", new_embedding)
    @test msg == "true"
    
    #verify that the record was updated
    row = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(collection_name) WHERE id_text = ?", ("record1",))))[1]
    @test occursin("4.0", row.embedding_json)  #simple substring check
    @test row.data_type == "Float64"  #assumes infer_data_type returns "Float64" for Float64 vectors
    
    # Test 2:Updating a non-existent record should throw an error
    @test_throws ErrorException WunDeeDB.update_embedding(db, collection_name, "nonexistent", new_embedding)
    
    #Test 3: Updating with an embedding of wrong dimension should throw an error
    wrong_embedding = [7.0, 8.0]  #length 2 (does not match meta length 3)
    @test_throws ErrorException WunDeeDB.update_embedding(db, collection_name, "record1", wrong_embedding)

    SQLite.close(db)
end

@testset "WunDeeDB.update_embedding Tests (db_path overload)" begin

    temp_dir = mktempdir()
    db_path = joinpath(temp_dir, "test_update_embedding.sqlite")
    collection_name = "test_update_embedding2"
    meta_table = "$(collection_name)_meta"


    db = SQLite.DB(db_path)
    SQLite.execute(db, "CREATE TABLE IF NOT EXISTS $(collection_name) (id_text TEXT, embedding_json TEXT, data_type TEXT)")
    SQLite.execute(db, """
    CREATE TABLE IF NOT EXISTS $(meta_table) (
         row_num BIGINT,
         vector_length INT
    )
    """)

    SQLite.execute(db, "INSERT INTO $(collection_name) (id_text, embedding_json, data_type) VALUES (?, ?, ?)", ("record2", "[10,20,30]", "Float64"))
    SQLite.execute(db, "INSERT INTO $(meta_table) (row_num, vector_length) VALUES (?, ?)", (1, 3))
    SQLite.close(db)


    new_embedding2 = [11.0, 22.0, 33.0]
    msg = WunDeeDB.update_embedding(db_path, collection_name, "record2", new_embedding2)
    @test msg == "true"


    db = SQLite.DB(db_path)
    row = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(collection_name) WHERE id_text = ?", ("record2",))))[1]
    @test occursin("11.0", row.embedding_json)
    SQLite.close(db)
    rm(temp_dir; force=true, recursive=true)
end


########
########.


@testset "WunDeeDB.bulk_update_embedding Tests (DB handle)" begin

    db = SQLite.DB()

    collection_name = "test_bulk_update"
    
    stmt_coll = """
    CREATE TABLE IF NOT EXISTS $(collection_name) (
         id_text TEXT,
         embedding_json TEXT,
         data_type TEXT
    )
    """
    SQLite.execute(db, stmt_coll)
    
    insert_stmt = "INSERT INTO $(collection_name) (id_text, embedding_json, data_type) VALUES (?, ?, ?)"
    SQLite.execute(db, insert_stmt, ("id1", "[1,2,3]", "Float64"))
    SQLite.execute(db, insert_stmt, ("id2", "[4,5,6]", "Float64"))
    SQLite.execute(db, insert_stmt, ("id3", "[7,8,9]", "Float64"))
    
    #prepare new embeddings to update
    id_texts = ["id1", "id2", "id3"]
    new_embeddings = [
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
        [16.0, 17.0, 18.0]
    ]
    
    #estt successful bulk update.
    msg = WunDeeDB.bulk_update_embedding(db, collection_name, id_texts, new_embeddings)
    @test msg == "true"
    
    #verify updates by querying the table.
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(collection_name)")))
    #sort rows by id_text for consistency.
    rows_sorted = sort(rows, by = r -> r.id_text)
    @test occursin("10.0", rows_sorted[1].embedding_json)
    @test occursin("13.0", rows_sorted[2].embedding_json)
    @test occursin("16.0", rows_sorted[3].embedding_json)
    @test all(r -> r.data_type == "Float64", rows_sorted)
    
    #test error: a mismatch between number of IDs and embeddings!
    id_texts_bad = ["id1", "id2"]
    @test_throws ErrorException WunDeeDB.bulk_update_embedding(db, collection_name, id_texts_bad, new_embeddings)
    
    # Test error: Inconsistent embedding lengths, no good length1!!!!!!!
    new_embeddings_bad = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0],       # Wrong length here!!!!!
        [7.0, 8.0, 9.0]
    ]
    @test_throws ErrorException WunDeeDB.bulk_update_embedding(db, collection_name, id_texts, new_embeddings_bad)
    
    #TODO get directly
    BULK_LIMIT=1000
    # Test error: Exceeding BULK_LIMIT.
    id_texts_over = [ "x$(i)" for i in 1:(BULK_LIMIT+1) ]
    new_embeddings_over = [ [1.0, 2.0, 3.0] for _ in 1:(BULK_LIMIT+1) ]
    @test_throws ErrorException WunDeeDB.bulk_update_embedding(db, collection_name, id_texts_over, new_embeddings_over)
    
    SQLite.close(db)
end

@testset "WunDeeDB.bulk_update_embedding Tests (db_path overload)" begin

    temp_dir = mktempdir()
    db_path = joinpath(temp_dir, "test_bulk_update.sqlite")
    collection_name = "test_bulk_update2"
    

    db = SQLite.DB(db_path)
    SQLite.execute(db, "CREATE TABLE IF NOT EXISTS $(collection_name) (id_text TEXT, embedding_json TEXT, data_type TEXT)")
    SQLite.close(db)
    

    db = SQLite.DB(db_path)
    for (id_val, json_val) in zip(["alpha", "beta"], ["[1,2,3]", "[4,5,6]"])
        SQLite.execute(db, "INSERT INTO $(collection_name) (id_text, embedding_json, data_type) VALUES (?, ?, ?)", (id_val, json_val, "Float64"))
    end
    SQLite.close(db)
    
    # update the records using the overload
    #use float embeddings so that infer_data_type returns Float64
    new_embeddings = [
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0]
    ]
    id_texts = ["alpha", "beta"]
    msg = WunDeeDB.bulk_update_embedding(db_path, collection_name, id_texts, new_embeddings)
    @test msg == "true"
    
    #reopen the database and verify!
    db = SQLite.DB(db_path)
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(collection_name)")))
    rows_sorted = sort(rows, by = r -> r.id_text)
    @test occursin("10.0", rows_sorted[1].embedding_json)
    @test occursin("13.0", rows_sorted[2].embedding_json)
    @test all(r -> r.data_type == "Float64", rows_sorted)
    
    SQLite.close(db)
    rm(temp_dir; force=true, recursive=true)
end


####################
####################


@testset "WunDeeDB.parse_data_type Tests" begin
    @test WunDeeDB.parse_data_type("Float64") === Float64
    @test_throws ErrorException WunDeeDB.parse_data_type("UnknownType")
end

@testset "WunDeeDB.get_embedding Tests (DB handle)" begin
    
    db = SQLite.DB()
    collection_name = "test_get_embedding"
    
    
    stmt = """
    CREATE TABLE IF NOT EXISTS $(collection_name) (
         id_text TEXT,
         embedding_json TEXT,
         data_type TEXT
    )
    """
    SQLite.execute(db, stmt)
    
    # Insert a record with a known embedding.
    insert_stmt = """
    INSERT INTO $(collection_name) (id_text, embedding_json, data_type)
    VALUES (?, ?, ?)
    """
    # for example, an embedding [1.1, 2.2, 3.3] encoded as JSON
    SQLite.execute(db, insert_stmt, ("record1", "[1.1,2.2,3.3]", "Float64"))
    
    #test: get_embedding should return the correct vector
    vec = WunDeeDB.get_embedding(db, collection_name, "record1")
    @test vec == [1.1, 2.2, 3.3]
    
    #test get_embedding for a non-existent record returns nothing
    vec2 = WunDeeDB.get_embedding(db, collection_name, "nonexistent")
    @test vec2 === nothing

    SQLite.close(db)
end

@testset "WunDeeDB.get_embedding Tests (db_path overload)" begin

    temp_dir = mktempdir()
    db_path = joinpath(temp_dir, "test_get_embedding.sqlite")
    collection_name = "test_get_embedding2"
    

    db = SQLite.DB(db_path)
    SQLite.execute(db, """
    CREATE TABLE IF NOT EXISTS $(collection_name) (
         id_text TEXT,
         embedding_json TEXT,
         data_type TEXT
    )
    """)
    #inserting a record
    SQLite.execute(db, """
    INSERT INTO $(collection_name) (id_text, embedding_json, data_type)
    VALUES (?, ?, ?)
    """, ("record2", "[4.4,5.5,6.6]", "Float64"))
    SQLite.close(db)
    
    #test the db_path overload
    result = WunDeeDB.get_embedding(db_path, collection_name, "record2")
    @test result == [4.4,5.5,6.6]
    
    # cleanup
    rm(temp_dir; force=true, recursive=true)
end


#################
#################



@testset "WunDeeDB.bulk_get_embedding Tests (DB handle)" begin
    db = SQLite.DB()
    collection_name = "test_bulk_get"
    
    
    stmt = """
    CREATE TABLE IF NOT EXISTS $(collection_name) (
         id_text TEXT,
         embedding_json TEXT,
         data_type TEXT
    )
    """
    SQLite.execute(db, stmt)
    
    
    insert_stmt = """
    INSERT INTO $(collection_name) (id_text, embedding_json, data_type)
    VALUES (?, ?, ?)
    """
    SQLite.execute(db, insert_stmt, ("r1", "[1.0,2.0,3.0]", "Float64"))
    SQLite.execute(db, insert_stmt, ("r2", "[4.0,5.0,6.0]", "Float64"))
    SQLite.execute(db, insert_stmt, ("r3", "[7.0,8.0,9.0]", "Float64"))
    
    #bulk_get_embedding should return a dictionary with keys for requested IDs
    result = WunDeeDB.bulk_get_embedding(db, collection_name, ["r1", "r3"])
    @test typeof(result) == Dict{String,Any}
    @test keys(result) == Set(["r1", "r3"])
    @test result["r1"] == [1.0,2.0,3.0]
    @test result["r3"] == [7.0,8.0,9.0]
    
    SQLite.close(db)
end

@testset "WunDeeDB.bulk_get_embedding Tests (db_path overload)" begin
    
    temp_dir = mktempdir()
    db_path = joinpath(temp_dir, "test_bulk_get.sqlite")
    collection_name = "test_bulk_get2"
    
    
    db = SQLite.DB(db_path)
    SQLite.execute(db, """
    CREATE TABLE IF NOT EXISTS $(collection_name) (
         id_text TEXT,
         embedding_json TEXT,
         data_type TEXT
    )
    """)
    
    SQLite.execute(db, """
    INSERT INTO $(collection_name) (id_text, embedding_json, data_type)
    VALUES (?, ?, ?)
    """, ("a", "[10.0,20.0,30.0]", "Float64"))
    SQLite.execute(db, """
    INSERT INTO $(collection_name) (id_text, embedding_json, data_type)
    VALUES (?, ?, ?)
    """, ("b", "[40.0,50.0,60.0]", "Float64"))
    SQLite.close(db)
    
    #yse the db_path overload
    result = WunDeeDB.bulk_get_embedding(db_path, collection_name, ["a", "b"])
    @test typeof(result) == Dict{String,Any}
    @test keys(result) == Set(["a", "b"])
    @test result["a"] == [10.0,20.0,30.0]
    @test result["b"] == [40.0,50.0,60.0]
    
    rm(temp_dir; force=true, recursive=true)
end



##############
##############


@testset "WunDeeDB.get_next_id Tests (DB handle)" begin
    db = SQLite.DB()
    collection_name = "test_get_next_id"
    

    stmt = """
    CREATE TABLE IF NOT EXISTS $(collection_name) (
         id_text TEXT,
         embedding_json TEXT,
         data_type TEXT
    )
    """
    SQLite.execute(db, stmt)
    
    insert_stmt = """
    INSERT INTO $(collection_name) (id_text, embedding_json, data_type)
    VALUES (?, ?, ?)
    """
    SQLite.execute(db, insert_stmt, ("a", "[1.0,2.0,3.0]", "Float64"))
    SQLite.execute(db, insert_stmt, ("b", "[4.0,5.0,6.0]", "Float64"))
    SQLite.execute(db, insert_stmt, ("c", "[7.0,8.0,9.0]", "Float64"))
    
    #test 1 with full_row == false
    next_id = WunDeeDB.get_next_id(db, collection_name, "a"; full_row=false)
    @test next_id == "b"
    
    next_id = WunDeeDB.get_next_id(db, collection_name, "b"; full_row=false)
    @test next_id == "c"
    
    next_id = WunDeeDB.get_next_id(db, collection_name, "c"; full_row=false)
    @test next_id === nothing
    
    #test 2: with full_row == true
    full_row = WunDeeDB.get_next_id(db, collection_name, "a"; full_row=true)
    @test full_row.id_text == "b"
    @test full_row.embedding == [4.0,5.0,6.0]
    @test full_row.data_type == "Float64"
    
    SQLite.close(db)
end

@testset "WunDeeDB.get_next_id Tests (db_path overload)" begin
    temp_dir = mktempdir()
    db_path = joinpath(temp_dir, "test_get_next_id.sqlite")
    collection_name = "test_get_next_id2"
    
    
    db = SQLite.DB(db_path)
    SQLite.execute(db, """
    CREATE TABLE IF NOT EXISTS $(collection_name) (
         id_text TEXT,
         embedding_json TEXT,
         data_type TEXT
    )
    """)
    
    SQLite.execute(db, "INSERT INTO $(collection_name) (id_text, embedding_json, data_type) VALUES (?, ?, ?)", ("x", "[10.0,20.0,30.0]", "Float64"))
    SQLite.execute(db, "INSERT INTO $(collection_name) (id_text, embedding_json, data_type) VALUES (?, ?, ?)", ("y", "[40.0,50.0,60.0]", "Float64"))
    SQLite.execute(db, "INSERT INTO $(collection_name) (id_text, embedding_json, data_type) VALUES (?, ?, ?)", ("z", "[70.0,80.0,90.0]", "Float64"))
    SQLite.close(db)
    
    
    result = WunDeeDB.get_next_id(db_path, collection_name, "x"; full_row=false)
    @test result == "y"
    
    result_full = WunDeeDB.get_next_id(db_path, collection_name, "x"; full_row=true)
    @test result_full.id_text == "y"
    @test result_full.embedding == [40.0,50.0,60.0]
    @test result_full.data_type == "Float64"
    
    #querying with a current_id that is the highest should return nothing
    result = WunDeeDB.get_next_id(db_path, collection_name, "z"; full_row=false)
    @test result === nothing
    
    rm(temp_dir; force=true, recursive=true)
end


################
################



@testset "WunDeeDB.get_previous_id Tests (DB handle)" begin
    db = SQLite.DB()
    collection_name = "test_get_previous_id"
    

    create_stmt = """
    CREATE TABLE IF NOT EXISTS $(collection_name) (
         id_text TEXT,
         embedding_json TEXT,
         data_type TEXT
    )
    """
    SQLite.execute(db, create_stmt)
    

    insert_stmt = """
    INSERT INTO $(collection_name) (id_text, embedding_json, data_type)
    VALUES (?, ?, ?)
    """
    SQLite.execute(db, insert_stmt, ("a", "[1.0,2.0,3.0]", "Float64"))
    SQLite.execute(db, insert_stmt, ("b", "[4.0,5.0,6.0]", "Float64"))
    SQLite.execute(db, insert_stmt, ("c", "[7.0,8.0,9.0]", "Float64"))
    
    #test 1 with full_row == false
    prev_id = WunDeeDB.get_previous_id(db, collection_name, "b"; full_row=false)
    @test prev_id == "a"
    
    prev_id = WunDeeDB.get_previous_id(db, collection_name, "c"; full_row=false)
    @test prev_id == "b"
    
    prev_id = WunDeeDB.get_previous_id(db, collection_name, "a"; full_row=false)
    @test prev_id === nothing
    
    #test 2: with full_row == true
    full_row = WunDeeDB.get_previous_id(db, collection_name, "c"; full_row=true)
    @test full_row.id_text == "b"
    @test full_row.embedding == [4.0,5.0,6.0]
    @test full_row.data_type == "Float64"
    
    SQLite.close(db)
end

@testset "WunDeeDB.get_previous_id Tests (db_path overload)" begin
    
    temp_dir = mktempdir()
    db_path = joinpath(temp_dir, "test_get_previous_id.sqlite")
    collection_name = "test_get_previous_id2"
    
    
    db = SQLite.DB(db_path)
    SQLite.execute(db, """
    CREATE TABLE IF NOT EXISTS $(collection_name) (
         id_text TEXT,
         embedding_json TEXT,
         data_type TEXT
    )
    """)
    
    SQLite.execute(db, "INSERT INTO $(collection_name) (id_text, embedding_json, data_type) VALUES (?, ?, ?)", ("x", "[10.0,20.0,30.0]", "Float64"))
    SQLite.execute(db, "INSERT INTO $(collection_name) (id_text, embedding_json, data_type) VALUES (?, ?, ?)", ("y", "[40.0,50.0,60.0]", "Float64"))
    SQLite.execute(db, "INSERT INTO $(collection_name) (id_text, embedding_json, data_type) VALUES (?, ?, ?)", ("z", "[70.0,80.0,90.0]", "Float64"))
    SQLite.close(db)
    
    #test db_path overload with full_row false
    result = WunDeeDB.get_previous_id(db_path, collection_name, "y"; full_row=false)
    @test result == "x"
    
    #the db_path overload with full_row true
    result_full = WunDeeDB.get_previous_id(db_path, collection_name, "z"; full_row=true)
    @test result_full.id_text == "y"
    @test result_full.embedding == [40.0,50.0,60.0]
    @test result_full.data_type == "Float64"
    
    #when no previous record exists
    result_none = WunDeeDB.get_previous_id(db_path, collection_name, "x"; full_row=false)
    @test result_none === nothing
    
    rm(temp_dir; force=true, recursive=true)
end

###########
###########


@testset "WunDeeDB.count_entries Tests (DB handle)" begin
    
    db = SQLite.DB()
    collection_name = "test_count_entries"
    meta_table = "$(collection_name)_meta"
    
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS $(collection_name) (
            id_text TEXT,
            embedding_json TEXT,
            data_type TEXT
        )
    """)
    
    
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS $(meta_table) (
            row_num BIGINT,
            vector_length INT
        )
    """)
    
    SQLite.execute(db, "INSERT INTO $(collection_name) (id_text, embedding_json, data_type) VALUES (?, ?, ?)", ("r1", "[1,2,3]", "Int32"))
    SQLite.execute(db, "INSERT INTO $(collection_name) (id_text, embedding_json, data_type) VALUES (?, ?, ?)", ("r2", "[4,5,6]", "Int32"))
    
    #count_entries should return 2
    count = WunDeeDB.count_entries(db, collection_name)
    @test count == 2
    
    #test the when update_meta is true, meta table is updated with row_num = count.
    #set meta table row to a dummy value
    SQLite.execute(db, "INSERT INTO $(meta_table) (row_num, vector_length) VALUES (?, ?)", (0, 3))
    count = WunDeeDB.count_entries(db, collection_name; update_meta=true)
    @test count == 2
    meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test meta_rows[1].row_num == 2  #updatedd meta information
    
    #test when collection is empty, count_entries returns 0 and clears meta
    SQLite.execute(db, "DELETE FROM $(collection_name)")
    count = WunDeeDB.count_entries(db, collection_name; update_meta=true)
    @test count == 0
    meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test meta_rows[1].row_num == 0
    @test meta_rows[1].vector_length === missing || isnothing(meta_rows[1].vector_length)
    
    SQLite.close(db)
end

@testset "WunDeeDB.count_entries Tests (db_path overload)" begin

    temp_dir = mktempdir()
    db_path = joinpath(temp_dir, "test_count_entries.sqlite")
    collection_name = "test_count_entries2"
    meta_table = "$(collection_name)_meta"
    
    db = SQLite.DB(db_path)
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS $(collection_name) (
            id_text TEXT,
            embedding_json TEXT,
            data_type TEXT
        )
    """)
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS $(meta_table) (
            row_num BIGINT,
            vector_length INT
        )
    """)
    
    
    SQLite.execute(db, "INSERT INTO $(collection_name) (id_text, embedding_json, data_type) VALUES (?, ?, ?)", ("x1", "[7,8,9]", "Int32"))
    SQLite.execute(db, "INSERT INTO $(collection_name) (id_text, embedding_json, data_type) VALUES (?, ?, ?)", ("x2", "[10,11,12]", "Int32"))
    SQLite.close(db)
    
    result = WunDeeDB.count_entries(db_path, collection_name)
    @test result == 2
    
    #insert a dummy meta row.
    db = SQLite.DB(db_path)
    SQLite.execute(db, "INSERT INTO $(meta_table) (row_num, vector_length) VALUES (?, ?)", (0, 3))
    SQLite.close(db)
    result = WunDeeDB.count_entries(db_path, collection_name; update_meta=true)
    @test result == 2
    #reopen the db to check meta table
    db = SQLite.DB(db_path)
    meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, "SELECT * FROM $(meta_table)")))
    @test meta_rows[1].row_num == 2
    SQLite.close(db)
    
    rm(temp_dir; force=true, recursive=true)
end



############
############



@testset "WunDeeDB.get_embedding_size Tests (DB handle)" begin
    db = SQLite.DB()
    collection_name = "test_embedding_size"
    meta_table = "$(collection_name)_meta"
    
    #create the meta table
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS $(meta_table) (
            row_num BIGINT,
            vector_length INT
        )
    """)
    
    #test 1 when no meta record exists, get_embedding_size should return 0
    size = WunDeeDB.get_embedding_size(db, collection_name)
    @test size == 0
    
    SQLite.execute(db, "INSERT INTO $(meta_table) (row_num, vector_length) VALUES (?, ?)", (5, 3))
    
    #test 2 now, get_embedding_size should return the vector length from the meta record
    size = WunDeeDB.get_embedding_size(db, collection_name)
    @test size == 3

    SQLite.close(db)
end

@testset "WunDeeDB.get_embedding_size Tests (db_path overload)" begin
    
    temp_dir = mktempdir()
    db_path = joinpath(temp_dir, "test_embedding_size.sqlite")
    collection_name = "test_embedding_size2"
    meta_table = "$(collection_name)_meta"
    
    
    db = SQLite.DB(db_path)
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS $(meta_table) (
            row_num BIGINT,
            vector_length INT
        )
    """)
    
    
    SQLite.execute(db, "INSERT INTO $(meta_table) (row_num, vector_length) VALUES (?, ?)", (10, 4))
    SQLite.close(db)
    
    result = WunDeeDB.get_embedding_size(db_path, collection_name)
    @test result == 4
    
    rm(temp_dir; force=true, recursive=true)
end


#############
#############


@testset "WunDeeDB.random_embeddings Tests (DB handle)" begin
    db = SQLite.DB()
    collection_name = "test_random_emb"
    
    create_stmt = """
    CREATE TABLE IF NOT EXISTS $(collection_name) (
         id_text TEXT,
         embedding_json TEXT,
         data_type TEXT
    )
    """
    SQLite.execute(db, create_stmt)
    
    #insert sample records
    insert_stmt = """
    INSERT INTO $(collection_name) (id_text, embedding_json, data_type)
    VALUES (?, ?, ?)
    """
    
    SQLite.execute(db, insert_stmt, ("r1", "[1.0,2.0,3.0]", "Float64"))
    SQLite.execute(db, insert_stmt, ("r2", "[4.0,5.0,6.0]", "Float64"))
    SQLite.execute(db, insert_stmt, ("r3", "[7.0,8.0,9.0]", "Float64"))
    
    #get 2 random embeddings
    results = WunDeeDB.random_embeddings(db, collection_name, 2)
    @test length(results) == 2
    for rec in results
        @test haskey(rec, "id_text")
        @test haskey(rec, "embedding")
        @test haskey(rec, "data_type")
        
        @test typeof(rec["embedding"]) <: AbstractVector
        @test length(rec["embedding"]) == 3
        @test rec["data_type"] == "Float64"
    end
    
    SQLite.close(db)
end

@testset "WunDeeDB.random_embeddings Tests (db_path overload)" begin
    
    temp_dir = mktempdir()
    db_path = joinpath(temp_dir, "test_random_emb.sqlite")
    collection_name = "test_random_emb2"
    
    #open the database and create the table
    db = SQLite.DB(db_path)
    SQLite.execute(db, """
    CREATE TABLE IF NOT EXISTS $(collection_name) (
         id_text TEXT,
         embedding_json TEXT,
         data_type TEXT
    )
    """)
    #insert two sample records.
    SQLite.execute(db, """
    INSERT INTO $(collection_name) (id_text, embedding_json, data_type)
    VALUES (?, ?, ?)
    """, ("a", "[10.0,20.0,30.0]", "Float64"))
    SQLite.execute(db, """
    INSERT INTO $(collection_name) (id_text, embedding_json, data_type)
    VALUES (?, ?, ?)
    """, ("b", "[40.0,50.0,60.0]", "Float64"))
    SQLite.close(db)
    
    #use the db_path overload to get 1 random embedding
    result = WunDeeDB.random_embeddings(db_path, collection_name, 1)
    @test length(result) == 1
    rec = result[1]
    @test haskey(rec, "id_text")
    @test haskey(rec, "embedding")
    @test haskey(rec, "data_type")
    @test typeof(rec["embedding"]) <: AbstractVector
    @test length(rec["embedding"]) == 3
    @test rec["data_type"] == "Float64"
    
    rm(temp_dir; force=true, recursive=true)
end
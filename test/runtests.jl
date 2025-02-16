using WunDeeDB
using Test
using SQLite



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


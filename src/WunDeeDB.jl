module WunDeeDB

using SQLite
using Tables, DBInterface
using DataFrames


export get_supported_data_types,
        initialize_db, open_db, close_db, delete_db, delete_all_embeddings,
        get_meta_data, update_description,
        infer_data_type,
        insert_embeddings,
        delete_embeddings,
        update_embeddings,
        get_embeddings,
        random_embeddings, 
        get_adjacent_id,
        count_entries

        

###################
#TODO:
#linear exact search for Retrieval (brute force)
#IVF, HSNW
###################

const DB_HANDLE = Ref{Union{SQLite.DB, Nothing}}(nothing) #handle for the db connection to use instead of open/close fast
const KEEP_DB_OPEN = Ref{Bool}(true) #keep open or not the db connection after use

#use 2 approaches for getting the endianness of the system
function check_is_little_endian()
    check1 = reinterpret(UInt8, [UInt16(1)])[1] == 1
    check2 = ntoh(UInt16(1)) != UInt16(1)

    if( check1 || check2 )
        return true
    else
        return false
    end
end

const IS_LITTLE_ENDIAN = check_is_little_endian() #boolian


const DATA_TYPE_MAP = Dict(
    "Float16"  => Float16,
    "Float32"  => Float32,
    "Float64"  => Float64,
    "Int8"     => Int8,
    "UInt8"    => UInt8,
    "Int16"    => Int16,
    "UInt16"   => UInt16,
    "Int32"    => Int32,
    "UInt32"   => UInt32,
    "Int64"    => Int64,
    "UInt64"   => UInt64,
    "Int128"   => Int128,
    "UInt128"  => UInt128
)



const MAIN_TABLE_NAME = "Embeddings"
const META_DATA_TABLE_NAME = "EmbeddingsMetaData"

const CREATE_MAIN_TABLE_STMT = """
        CREATE TABLE IF NOT EXISTS $(MAIN_TABLE_NAME) (
            id_text TEXT PRIMARY KEY,
            embedding_blob BLOB NOT NULL
        )
        """

const CREATE_META_TABLE_STMT = """
       CREATE TABLE IF NOT EXISTS $(META_DATA_TABLE_NAME) (
	   embedding_count BIGINT,
	   embedding_length INT,
	   data_type TEXT NOT NULL,
       endianness TEXT NOT NULL,
       description TEXT
       )
       """

const DELETE_EMBEDDINGS_STMT = "DELETE FROM $(MAIN_TABLE_NAME)"

const META_TABLE_FULL_ROW_INSERTION_STMT = """
        INSERT INTO $(META_DATA_TABLE_NAME) (embedding_count, embedding_length, data_type, endianness, description)
        VALUES (?, ?, ?, ?, ?)
        """

const META_SELECT_ALL_QUERY = "SELECT * FROM $(META_DATA_TABLE_NAME)"
const META_UPDATE_QUERY = "UPDATE $(META_DATA_TABLE_NAME) SET embedding_count = ?"
const META_UPDATE_DESCRIPTION = "UPDATE $(META_DATA_TABLE_NAME) SET description = ?"
const META_RESET_STMT = "UPDATE $(META_DATA_TABLE_NAME) SET embedding_count = 0"

const INSERT_EMBEDDING_STMT = "INSERT INTO $MAIN_TABLE_NAME (id_text, embedding_blob) VALUES (?, ?)"


"""
    get_supported_data_types() -> Vector{String}

Returns a sorted vector of supported data type names as strings.
"""
function get_supported_data_types()::Vector{String}
    return sort(collect(keys(DATA_TYPE_MAP)))
end





"""
Initialize a SQLite database by setting up the main and meta tables with appropriate configuration.

# Arguments
- `db_path::String`: Path to the SQLite database file.
- `embedding_length::Int`: Length of the embedding vector. Must be 1 or greater.
- `data_type::String`: Data type for the embeddings. Must be one of the supported types (use `get_supported_data_types()` to see valid options).
- `description::String=""`: (optional) User selected description meta data, defaults to empty string
- `keep_conn_open::Bool=true`: (optional) Keep the DB connection open for rapid successive uses or false for multiple applications to release

# Returns
- `true` on successful initialization.
- A `String` error message if any parameter is invalid or if an exception occurs during initialization.

# Example
```julia
result = initialize_db("my_database.db", 128, "float32", description="embeddings from 01/01/25", keep_conn_open=true)
if result === true
    println("Database initialized successfully!")
else
    println("Initialization failed: 'result'")
end

"""
function initialize_db(db_path::String, embedding_length::Int, data_type::String; description::String="", keep_conn_open::Bool=true)

    keep_conn_open ? KEEP_DB_OPEN[] = true : KEEP_DB_OPEN[] = false

    endianness = "small"

    arg_support_string = ""

    if !(data_type in keys(DATA_TYPE_MAP))
        arg_support_string *= "Unsupported data_type, run get_supported_data_types() to get the supported types. "
    end

    if embedding_length < 1
        arg_support_string *= "Embedding_length must be 1 or greater. "
    end

    if length(arg_support_string) > 0
        return arg_support_string
    end

    db = open_db(db_path)
    
    try
        SQLite.execute(db, "PRAGMA journal_mode = WAL;")
        SQLite.execute(db, "PRAGMA synchronous = NORMAL;")

        SQLite.transaction(db) do
            SQLite.execute(db, CREATE_MAIN_TABLE_STMT)
            SQLite.execute(db, CREATE_META_TABLE_STMT)

            rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, META_SELECT_ALL_QUERY)))

            if isempty(rows) #if empty, insert initial meta information
                #in initial stage set embedding_count to 0 because no embeddings have been added yet
                SQLite.execute(db, META_TABLE_FULL_ROW_INSERTION_STMT, (0, embedding_length, data_type, endianness, description))
            end
        end

        if KEEP_DB_OPEN[] 
            DB_HANDLE[] = db
        else
            close_db(db)
        end

        return true
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end


"""
open_db(db_path::String) -> SQLite.DB

Open an SQLite database located at the specified file path, ensuring that the directory exists.

This function performs the following steps:
1. **Directory Check and Creation:**  
   It determines the directory path for `db_path` and checks whether it exists. If the directory does not exist, the function attempts to create it using `mkpath`.  
   If directory creation fails, an error is raised with a descriptive message.

2. **Database Connection and Configuration:**  
   The function opens an SQLite database connection using `SQLite.DB(db_path)`.  
   It then sets two PRAGMA options for improved write performance:
   - `journal_mode` is set to `WAL` (Write-Ahead Logging).
   - `synchronous` is set to `NORMAL`.

3. **Return Value:**  
   The configured SQLite database connection is returned.

# Arguments
- `db_path::String`: The file path to the SQLite database. The function will ensure that the directory containing this file exists
- `keep_conn_open::Bool`: Optional, whether the connection should persist on the session after the function returns

# Returns
- An instance of `SQLite.DB` representing the open and configured database connection.

# Example
```julia
db = open_db("data/mydatabase.sqlite", keep_conn_open="true")
# Use the database connection...
SQLite.execute(db, "SELECT * FROM my_table;")
# Don't forget to close the database when done.
close_db(db)
"""
function open_db(db_path::String; keep_conn_open::Bool=true)

    KEEP_DB_OPEN[] = keep_conn_open

    if keep_conn_open
        if isnothing(DB_HANDLE[]) == false
            return DB_HANDLE[]
        else
            dirpath = Base.dirname(db_path)
            if !isdir(dirpath)
                try
                    mkpath(dirpath)
                catch e
                    error("ERROR: Could not create directory $dirpath. Original error: $(e)")
                end
            end
            
            db = SQLite.DB(db_path)
            SQLite.execute(db, "PRAGMA journal_mode = WAL;")
            SQLite.execute(db, "PRAGMA synchronous = NORMAL;")
            DB_HANDLE[] = db
            return db
        end
    else
        DB_HANDLE[] = nothing    
        dirpath = Base.dirname(db_path)
        
        if !isdir(dirpath)
            try
                mkpath(dirpath)
            catch e
                error("ERROR: Could not create directory $dirpath. Original error: $(e)")
            end
        end
        
        db = SQLite.DB(db_path)
        SQLite.execute(db, "PRAGMA journal_mode = WAL;")
        SQLite.execute(db, "PRAGMA synchronous = NORMAL;")
        return db
    end
end


""" close_db(db::SQLite.DB)

Close an open SQLite database connection.

This function is a simple wrapper around SQLite.close to ensure that the provided database connection is properly closed when it is no longer needed.

# Arguments

- db::SQLite.DB: (optional) The SQLite database connection to be closed, and if not included the default of the persistent db object is used

# Example
```julia
db = open_db("data/mydatabase.sqlite")
# Perform database operations...
close_db(db)

or 
close_db()
""" 
function close_db(db::SQLite.DB)
    SQLite.close(db)
    DB_HANDLE[] = nothing
    KEEP_DB_OPEN[] = false
end

function close_db()
    if !isnothing(DB_HANDLE[])
        SQLite.close(DB_HANDLE[])
    end
    DB_HANDLE[] = nothing
    KEEP_DB_OPEN[] = false
end



"""
Delete the database file at the specified path.

# Arguments
- `db_path::String`: The file path of the database to delete.

# Returns
- `true` if the file was successfully deleted.
- A `String` error message if deletion fails or if the file does not exist.

# Example
```julia
result = delete_db("my_database.db")
if result === true
    println("Database deleted successfully.")
else
    println("Error: 'result'")
end

"""
function delete_db(db_path::String)

    if !isnothing(DB_HANDLE[])
        SQLite.close(DB_HANDLE[])
    end

    KEEP_DB_OPEN[] = false
    DB_HANDLE[] = nothing

    if isfile(db_path)
        try
            rm(db_path)
            return true
        catch e
            return "Error deleting DB: $(e)"
        end
    else
        return "Database file does not exist."
    end
end

"""
Delete all embeddings from the database at the specified path and reset the embedding count.

# Arguments
- `db_path::String`: The file path of the SQLite database.

# Returns
- `true` if the operation is successful.
- A `String` error message if an error occurs.

# Example
```julia
result = delete_all_embeddings("my_database.db")
if result === true
    println("Embeddings deleted successfully.")
else
    println("Error: 'result'")
end

"""
function delete_all_embeddings(db_path::String)
    db = !isnothing(DB_HANDLE[]) ? DB_HANDLE[] : open_db(db_path)
    
    try
        SQLite.transaction(db) do
            SQLite.execute(db, DELETE_EMBEDDINGS_STMT) #clear all embeddings
            SQLite.execute(db, META_RESET_STMT) #reset the embedding count to 0
        end

        return true
    catch e
        close_db(db)
        DB_HANDLE[] = nothing
        KEEP_DB_OPEN[] = false
        return "Error: $(e)"
    end
end



"""
Retrieve meta data from the SQLite database.

This function is overloaded to accept either an active database connection or a file path:
- `get_meta_data(db::SQLite.DB)`: Retrieves the meta data from the given open database connection.
- `get_meta_data(db_path::String)`: Opens the database at the specified path, retrieves the meta data, and then closes the connection.
- neither approach will close the DB if a persistent handle is in use and tries to use the persistent one if possible.

# Arguments
- For `get_meta_data(db::SQLite.DB)`:
  - `db::SQLite.DB`: An active SQLite database connection.
- For `get_meta_data(db_path::String)`:
  - `db_path::String`: The file path to the SQLite database.

# Returns
- The first row of meta data as a named tuple if it exists, or `nothing` if no meta data is found.
- If an error occurs (in the `db_path` overload), a `String` error message is returned.

# Examples

Using an existing database connection:
```julia
meta = get_meta_data(db)
if meta !== nothing
    println("Meta data: ", meta)
else
    println("No meta data available.")
end

Using a database file path:

result = get_meta_data("my_database.db")
if result isa NamedTuple
    println("Meta data: ", result)
else
    println("Error: ", result)
end

"""
function get_meta_data(db::SQLite.DB)
    stmt = "SELECT * FROM $META_DATA_TABLE_NAME"
    df = DBInterface.execute(db, stmt) |> DataFrame

    if isempty(df)
        return nothing
    else
        return df
    end
end

function get_meta_data(db_path::String)
    # db = open_db(db_path)
    db = !isnothing(DB_HANDLE[]) ? DB_HANDLE[] : open_db(db_path)

    try
        result = get_meta_data(db)
        return result
    catch e
        close_db(db)
        DB_HANDLE[] = nothing
        KEEP_DB_OPEN[] = false
        return "Error: $(e)"
    end
end



"""
Update the description in the metadata table.

This function can be called in two ways:
1. With an open SQLite.DB connection.
2. With a database file path, which will open the connection if needed.

It executes a parameterized query to update the description field. If an error occurs,
the function closes the database connection, resets global connection variables, and returns an error message.

Arguments:
- `db::SQLite.DB` or `db_path::String`: A SQLite database connection or the path to the database file.
- `description::String` (optional): The new description to set (defaults to an empty string).

Returns:
- true on success, or a string with the error message on failure
"""
function update_description(db::SQLite.DB, description::String="")
    try
        DBInterface.execute(db, META_UPDATE_DESCRIPTION, (description,)) #SQLite.execute(db, META_UPDATE_DESCRIPTION, (description,))
    catch e
        close_db(db)
        DB_HANDLE[] = nothing
        KEEP_DB_OPEN[] = false
        return "Error: $(e)"
    end

    return true
end
function update_description(db_path::String, description::String="")
    db = !isnothing(DB_HANDLE[]) ? DB_HANDLE[] : open_db(db_path)

    try
        DBInterface.execute(db, META_UPDATE_DESCRIPTION, (description,)) #SQLite.execute(db, META_UPDATE_DESCRIPTION, (description,))
    catch e
        close_db(db)
        DB_HANDLE[] = nothing
        KEEP_DB_OPEN[] = false
        return "Error: $(e)"
    end
end



function update_meta(db::SQLite.DB, count::Int=1)
    #rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, META_SELECT_ALL_QUERY)))
    df = DBInterface.execute(db, META_SELECT_ALL_QUERY) |> DataFrame
    
    if isempty(df)
        error("Meta data table is empty. The meta row should have been initialized during database setup.")
    else
        row = df[1,:]
        current_count = row.embedding_count
        new_count = current_count + count
        
        DBInterface.execute(db, META_UPDATE_QUERY, (new_count,))
    end

    return true
end






"""
Infer the data type of the elements in a numeric embedding vector.

# Arguments
- `embedding::AbstractVector{<:Number}`: A vector containing numerical values.

# Returns
- A `String` representing the element type of the embedding.

# Example
```julia
vec = [1.0, 2.0, 3.0]
println(infer_data_type(vec))  # "Float64"
"""
function infer_data_type(embedding::AbstractVector{<:Number})
    return string(eltype(embedding))
end


# function embedding_to_blob(embedding::AbstractVector{<:Number})
#     return Vector{UInt8}(reinterpret(UInt8, embedding))
# end

function embedding_to_blob(embedding::AbstractVector{T}) where T <: Number
    if IS_LITTLE_ENDIAN
        return Vector{UInt8}(reinterpret(UInt8, embedding))
    else
        # If T occupies only one byte, no swap is necessary.
        if sizeof(T) == 1
            return Vector{UInt8}(reinterpret(UInt8, embedding))
        end
        swapped = if T <: Integer
            bswap.(embedding)
        elseif T == Float16
            bswap.(reinterpret(UInt16, embedding))
        elseif T == Float32
            bswap.(reinterpret(UInt32, embedding))
        elseif T == Float64
            bswap.(reinterpret(UInt64, embedding))
        elseif T in (Int128, UInt128)
            bswap.(embedding)
        else
            error("Type $T not supported for byte swapping")
        end
        return Vector{UInt8}(reinterpret(UInt8, swapped))
    end
end






"""
Insert one or more embeddings into a specified collection in the SQLite database.

This function is overloaded to support:
- **Active Connection**: `insert_embeddings(db::SQLite.DB, id_input, embedding_input)`
- **Database Path**: `insert_embeddings(db_path::String, id_input, embedding_input)`

In both cases, the function validates that the provided embeddings have a consistent length and that their data type matches the meta information stored in the database. For the method accepting a database path, the connection is automatically opened and closed.

# Arguments
- `db::SQLite.DB` or `db_path::String`: Either an active SQLite database connection or the file path to the database.
- `id_input`: A single ID or an array of IDs corresponding to the embeddings.
- `embedding_input`: A single numeric embedding vector or an array of embedding vectors. All embeddings must be of the same length.

# Returns
- `true` if the embeddings are successfully inserted.
- A `String` error message if an error occurs.

# Examples

Using an active database connection:
```julia
result = insert_embeddings(db, [1, 2], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
if result === true
    println("Embeddings inserted successfully.")
else
    println("Error: ", result)
end

Using a database file path:

result = insert_embeddings("my_database.db", 1, [0.1, 0.2, 0.3])
if result === true
    println("Embedding inserted successfully.")
else
    println("Error: ", result)
end
"""
function insert_embeddings(db::SQLite.DB, id_input, embedding_input)
    #if a single ID or embedding is passed, wrap it in a one-element array
    ids = id_input isa AbstractVector ? id_input : [id_input]
    
    embeddings = if embedding_input isa AbstractVector{<:Number} && !(embedding_input isa AbstractVector{<:AbstractVector})
        [embedding_input]
    elseif embedding_input isa AbstractVector{<:AbstractVector}
        embedding_input
    else
        error("Invalid type for embedding_input.")
    end

    n = length(ids)
    if n != length(embeddings)
        error("Mismatch between number of IDs and embeddings")
    end

    #see all embeddings have the same length
    emb_length = length(embeddings[1])
    for e in embeddings
        if length(e) != emb_length
            error("All embeddings must have the same length")
        end
    end

    local inferred_data_type = infer_data_type(embeddings[1]) #see data type from the first embedding

    #retrieve meta table information using DBInterface.execute with Tables.namedtupleiterator
    meta_df = DBInterface.execute(db, META_SELECT_ALL_QUERY) |> DataFrame #collect(Tables.namedtupleiterator(DBInterface.execute(db, META_SELECT_ALL_QUERY)))
    
    if isempty(meta_df)
        error("Meta table is empty. The meta row should have been initialized during database setup.")
    end
    
    meta = meta_df[1,:]
    
    if meta.embedding_length != emb_length
        error("Embedding length mismatch: meta table has embedding_length=$(meta.embedding_length) but new embeddings have length=$(emb_length)")
    end
    if meta.data_type != inferred_data_type
        error("Data type mismatch: meta table has data_type=$(meta.data_type) but new embeddings are of type=$(inferred_data_type)")
    end

    #get the embedding blob by build the db parameters for each row
    params = [(string(ids[i]), embedding_to_blob(embeddings[i])) for i in 1:n]

    #embedding insertion and meta update within a transaction.
    SQLite.transaction(db) do
        for p in params
            DBInterface.execute(db, INSERT_EMBEDDING_STMT, p) #SQLite.execute(db, INSERT_EMBEDDING_STMT, p)
        end
        #update the meta table by incrementing embedding_count by n
        update_meta(db, n)
    end

    return true
end

function insert_embeddings(db_path::String, id_input, embedding_input)
    db = !isnothing(DB_HANDLE[]) ? DB_HANDLE[] : open_db(db_path)

    try 
        msg = insert_embeddings(db, id_input, embedding_input)
        return msg
    catch e
        close_db(db)
        DB_HANDLE[] = nothing
        KEEP_DB_OPEN[] = false
        return "Error: $(e)"
    end
end



"""
Delete one or more embeddings from the database using their ID(s).

This function is overloaded to support both an active database connection and a database file path:
- `delete_embeddings(db::SQLite.DB, id_input)`: Deletes embeddings using an open database connection.
- `delete_embeddings(db_path::String, id_input)`: Opens the database at the specified path, deletes the embeddings, and then closes the connection.

# Arguments
- For `delete_embeddings(db::SQLite.DB, id_input)`:
  - `db::SQLite.DB`: An active SQLite database connection.
  - `id_input`: A single ID or a collection of IDs (can be any type convertible to a string) identifying the embeddings to be deleted.
- For `delete_embeddings(db_path::String, id_input)`:
  - `db_path::String`: The file path to the SQLite database.
  - `id_input`: A single ID or a collection of IDs identifying the embeddings to be deleted.

# Returns
- `true` if the deletion is successful.
- A `String` error message if an error occurs during deletion.

# Examples

Using an active database connection:
```julia
result = delete_embeddings(db, [1, 2, 3])
if result === true
    println("Embeddings deleted successfully.")
else
    println("Error: ", result)
end

Using a database file path:

result = delete_embeddings("my_database.db", 1)
if result === true
    println("Embedding deleted successfully.")
else
    println("Error: ", result)
end

"""
function delete_embeddings(db::SQLite.DB, id_input)
    #a single ID is passed, wrap it in a one-element array
    ids = id_input isa AbstractVector ? map(string, id_input) : [string(id_input)]
    n = length(ids)
    if n == 0
        error("No IDs provided for deletion.")
    end

    SQLite.transaction(db) do
        #build comma-separated placeholders based on the number of IDs
        placeholders = join(fill("?", n), ", ")
        stmt = "DELETE FROM $MAIN_TABLE_NAME WHERE id_text IN ($placeholders)"
        params = Tuple(ids)
        DBInterface.execute(db, stmt, params)
        
        #update the meta table for bulk deletion: subtract 'n' from the meta embedding_count
        update_meta(db, -n)
    end

    return true
end

function delete_embeddings(db_path::String, id_input)
    #check for an empty vector
    if (id_input isa AbstractVector) && isempty(id_input)
        return "Error: No IDs provided for deletion."
    end

    db = !isnothing(DB_HANDLE[]) ? DB_HANDLE[] : open_db(db_path)
    
    try
        result = delete_embeddings(db, id_input)
        return result
    catch e
        close_db(db)
        DB_HANDLE[] = nothing
        KEEP_DB_OPEN[] = false
        return "Error: $(e)"
    end
end





"""
Update one or more embeddings in the SQLite database with new embedding data.

This function is overloaded to support two usage patterns:
- `update_embeddings(db::SQLite.DB, id_input, new_embedding_input)`: Updates embeddings using an active database connection.
- `update_embeddings(db_path::String, id_input, new_embedding_input)`: Opens the database at the specified path, updates the embeddings, and then closes the connection.

The function accepts a single identifier or an array of identifiers along with corresponding new embedding vectors. It validates that all new embeddings have the same length, and that their length and data type match the values stored in the meta table. For single record updates, it additionally confirms that the record exists in the database.

# Arguments
- `db::SQLite.DB` or `db_path::String`: Either an active database connection or the file path to the SQLite database.
- `id_input`: A single ID or an array of IDs identifying the embeddings to update.
- `new_embedding_input`: A single numeric embedding vector or an array of such vectors. All embeddings must be of consistent length.

# Returns
- `true` if the update is successful.
- A `String` error message if an error occurs.

# Examples

Using an active database connection:
```julia
result = update_embeddings(db, 1, [0.5, 0.6, 0.7])
if result === true
    println("Embedding updated successfully.")
else
    println("Error: ", result)
end

Using a database file path:

result = update_embeddings("my_database.db", [1, 2], [[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
if result === true
    println("Embeddings updated successfully.")
else
    println("Error: ", result)
end

"""
# XXX df = DBInterface.execute(db, stmt) |> DataFrame
function update_embeddings(db::SQLite.DB, id_input, new_embedding_input)
    #wrap a single ID or embedding into a one-element array
    ids = id_input isa AbstractVector ? id_input : [id_input]
    
    new_embeddings = if new_embedding_input isa AbstractVector{<:Number} && !(new_embedding_input isa AbstractVector{<:AbstractVector})
        [new_embedding_input]
    elseif new_embedding_input isa AbstractVector{<:AbstractVector}
        new_embedding_input
    else
        error("Invalid type for new_embedding_input.")
    end

    n = length(ids)
    if n != length(new_embeddings)
        error("Mismatch between number of IDs and new embeddings.")
    end

    #ensure all new embeddings have the same length
    new_emb_length = length(new_embeddings[1])
    for emb in new_embeddings
        if length(emb) != new_emb_length
            error("All new embeddings must have the same length.")
        end
    end

    #infer data type from the first new embedding
    local inferred_data_type = infer_data_type(new_embeddings[1])

    #get meta table information using the Tables interface
    meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, META_SELECT_ALL_QUERY)))
    if isempty(meta_rows)
        error("Meta table is empty. The meta row should have been initialized during database setup.")
    end
    meta = meta_rows[1]
    
    #new embedding dimensions and data_type also match the meta table?
    if meta.embedding_length != new_emb_length
        error("Embedding length mismatch: meta table has embedding_length=$(meta.embedding_length) but new embeddings have length=$(new_emb_length).")
    end
    if meta.data_type != inferred_data_type
        error("Data type mismatch: meta table has data_type=$(meta.data_type) but new embeddings are of type=$(inferred_data_type).")
    end

    #single update: check that the record exists
    if n == 1
        check_sql = "SELECT 1 AS found FROM $MAIN_TABLE_NAME WHERE id_text = ?" # TODO: make global constant

        found_rows = SQLite.transaction(db) do
            collect(Tables.namedtupleiterator(DBInterface.execute(db, check_sql, (string(ids[1]),))))
        end

        if isempty(found_rows)
            error("Record with id $(ids[1]) not found.")
        end
    end

    #the update statement (using binary blobs)
    local update_stmt = "UPDATE $MAIN_TABLE_NAME SET embedding_blob = ? WHERE id_text = ?" # TODO: make global constant
    
    #execute all updates within a transactio
    SQLite.transaction(db) do
        for i in 1:n
            local blob = embedding_to_blob(new_embeddings[i])
            SQLite.execute(db, update_stmt, (blob, string(ids[i])))
        end
    end

    return true
end

function update_embeddings(db_path::String, id_input, new_embedding_input)
    db = !isnothing(DB_HANDLE[]) ? DB_HANDLE[] : open_db(db_path)
    
    try
        result = update_embeddings(db, id_input, new_embedding_input)

        return result
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end

















#RETRIEVE

function parse_data_type(dt::String) 
    T = get(DATA_TYPE_MAP, dt, nothing) 
    if T === nothing 
        error("Unsupported data type: $dt") 
    end 
    
    return T 
end

function blob_to_embedding(blob::Vector{UInt8}, ::Type{T}) where T
    if IS_LITTLE_ENDIAN
        return collect(reinterpret(T, blob))
    else
        # If the system is big-endian, swap the bytes back.
        # Only perform swap if T occupies more than one byte.
        if sizeof(T) == 1
            return collect(reinterpret(T, blob))
        else
            swapped = if T <: Integer
                bswap.(reinterpret(T, blob))
            elseif T == Float16
                bswap.(reinterpret(UInt16, blob))
            elseif T == Float32
                bswap.(reinterpret(UInt32, blob))
            elseif T == Float64
                bswap.(reinterpret(UInt64, blob))
            elseif T in (Int128, UInt128)
                bswap.(reinterpret(T, blob))
            else
                error("Type $T not supported for byte swapping")
            end
            return collect(reinterpret(T, reinterpret(UInt8, swapped)))
        end
    end
end



"""
Retrieve one or more embeddings from the SQLite database by their ID(s).

This function is overloaded to support:
- `get_embeddings(db::SQLite.DB, id_input)`: Retrieves embeddings using an active database connection.
- `get_embeddings(db_path::String, id_input)`: Opens the database at the specified path, retrieves embeddings, and then closes the connection.

When a single ID is provided, the corresponding embedding vector is returned (or `nothing` if not found). When multiple IDs are provided, a dictionary mapping each ID (as a string) to its embedding vector is returned.

# Arguments
- For `get_embeddings(db::SQLite.DB, id_input)`:
  - `db::SQLite.DB`: An active SQLite database connection.
  - `id_input`: A single ID or an array of IDs identifying the embeddings to retrieve.
- For `get_embeddings(db_path::String, id_input)`:
  - `db_path::String`: The file path to the SQLite database.
  - `id_input`: A single ID or an array of IDs identifying the embeddings to retrieve.

# Returns
- A single embedding vector if one ID is provided, or a dictionary mapping IDs (as strings) to embedding vectors if multiple IDs are provided.
- Returns `nothing` if a single requested ID is not found.
- A `String` error message if an error occurs during retrieval.

# Examples

Using an active database connection:
```julia
embedding = get_embeddings(db, 42)
if embedding === nothing
    println("Embedding not found.")
else
    println("Embedding: ", embedding)
end

Using a database file path:

embeddings = get_embeddings("my_database.db", [1, 2, 3])
for (id, emb) in embeddings
    println("ID: 'id', Embedding: ", emb)
end

"""
function get_embeddings(db::SQLite.DB, id_input)
    #wrap a single ID into an array
    ids = id_input isa AbstractVector ? id_input : [id_input]
    n = length(ids)
    if n == 0
        error("No IDs provided for retrieval.")
    end

    #build the query using an IN clause with comma-separated placeholders
    placeholders = join(fill("?", n), ", ")
    stmt = "SELECT id_text, embedding_blob FROM $MAIN_TABLE_NAME WHERE id_text IN ($placeholders)"
    params = Tuple(string.(ids))
    
    #get rows from the main table
    rows = SQLite.transaction(db) do
        collect(Tables.namedtupleiterator(DBInterface.execute(db, stmt, params)))
    end
    
    #get meta table information to determine the stored data type
    meta_rows = SQLite.transaction(db) do
        collect(Tables.namedtupleiterator(DBInterface.execute(db, META_SELECT_ALL_QUERY)))
    end

    if isempty(meta_rows)
        error("Meta table is empty. The meta row should have been initialized during database setup.")
    end
    meta = meta_rows[1]
    dt_string = meta.data_type
    T = parse_data_type(dt_string) #TODO: keep as a persistant global variable
    
    #make a dictionary mapping id_text to its embedding vector
    result = Dict{String,Any}()
    for row in rows
        id = row.id_text
        blob = row.embedding_blob  # this is a Vector{UInt8}
        embedding_vec = blob_to_embedding(blob, T)
        result[string(id)] = embedding_vec
    end
    
    #only one ID was requested, return its embedding directly (or nothing if not found)
    if n == 1
        return isempty(result) ? nothing : first(values(result))
    else
        return result
    end
end

function get_embeddings(db_path::String, id_input)
    db = !isnothing(DB_HANDLE[]) ? DB_HANDLE[] : open_db(db_path)

    try
        result = get_embeddings(db, id_input)

        return result
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end



"""
Randomly retrieve a specified number of embeddings from the SQLite database.

This function is overloaded to support two usage patterns:
- `random_embeddings(db::SQLite.DB, num::Int)`: Retrieves embeddings using an active database connection.
- `random_embeddings(db_path::String, num::Int)`: Opens the database at the specified path, retrieves embeddings, and then closes the connection.

# Arguments
- `db::SQLite.DB` or `db_path::String`: Either an active SQLite database connection or the file path to the SQLite database.
- `num::Int`: The number of random embeddings to retrieve.

# Returns
- A `Dict{String, Any}` mapping each embedding's ID (as a string) to its embedding vector.

# Example
```julia
embeddings = random_embeddings("my_database.db", 5)
for (id, emb) in embeddings
    println("ID: 'id', Embedding: ", emb)
end

""" 
function random_embeddings(db::SQLite.DB, num::Int)
    # TODO: global stmt
    stmt = """
        SELECT id_text, embedding_blob
        FROM $MAIN_TABLE_NAME
        ORDER BY RANDOM()
        LIMIT ?;
    """
    rows = SQLite.transaction(db) do
        collect(Tables.namedtupleiterator(DBInterface.execute(db, stmt, (num,))))
    end
    
    #meta table info to determine the stored data type
    meta_rows = SQLite.transaction(db) do
        collect(Tables.namedtupleiterator(DBInterface.execute(db, META_SELECT_ALL_QUERY)))
    end

    if isempty(meta_rows)
        error("Meta table is empty. The meta row should have been initialized during database setup.")
    end
    meta = meta_rows[1]
    T = parse_data_type(meta.data_type)
    
    #make the result dictionary mapping id_text to the embedding vector
    result = Dict{String,Any}()
    for row in rows
        embedding_vec = blob_to_embedding(row.embedding_blob, T)
        result[string(row.id_text)] = embedding_vec
    end
    return result
end

function random_embeddings(db_path::String, num::Int)
    db = !isnothing(DB_HANDLE[]) ? DB_HANDLE[] : open_db(db_path)

    try
        result = random_embeddings(db, num)

        return result
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end





"""
Retrieve the adjacent record relative to a given `current_id` from the SQLite database.

This function is overloaded to support both an active database connection and a database file path:
- `get_adjacent_id(db::SQLite.DB, current_id; direction="next", full_row=true)`: Uses an active connection.
- `get_adjacent_id(db_path::String, current_id; direction="next", full_row=true)`: Opens the database at the specified path, retrieves the adjacent record, and then closes the connection.

The function returns the record immediately after (or before) the specified `current_id` based on the `direction` parameter. When `full_row` is `true`, the returned result is a named tuple containing the `id_text`, the decoded embedding vector, and the stored `data_type`. When `full_row` is `false`, only the `id_text` is returned.

# Arguments
- **For `get_adjacent_id(db::SQLite.DB, current_id; direction, full_row)`**:
  - `db::SQLite.DB`: An active SQLite database connection.
  - `current_id`: The current record's ID from which to find the adjacent record.
  - `direction::String="next"`: The direction to search for the adjacent record. Use `"next"` for the record with an ID greater than `current_id`, or `"previous"` (or `"prev"`) for the record with an ID less than `current_id`.
  - `full_row::Bool=true`: If `true`, return the full record (including embedding and meta data); if `false`, return only the `id_text`.

- **For `get_adjacent_id(db_path::String, current_id; direction, full_row)`**:
  - `db_path::String`: The file path to the SQLite database.
  - Other parameters are as described above.

# Returns
- When `full_row` is `true`: A named tuple `(id_text, embedding, data_type)` representing the adjacent record.
- When `full_row` is `false`: The adjacent record's `id_text`.
- Returns `nothing` if no adjacent record is found.
- For the `db_path` overload, a `String` error message is returned if an error occurs.

# Example
```julia
# Using an active database connection:
adjacent = get_adjacent_id(db, 100; direction="previous", full_row=false)
if adjacent !== nothing
    println("Adjacent ID: ", adjacent)
else
    println("No adjacent record found.")
end

# Using a database file path:
result = get_adjacent_id("my_database.db", 100; direction="next")
if result isa NamedTuple
    println("Adjacent record: ", result)
else
    println("Error or record not found: ", result)
end

"""
function get_adjacent_id(db::SQLite.DB, current_id; direction="next", full_row=true)
    
    if direction == "next"
        comp_op = ">"
        order_clause = "ASC"
    elseif direction == "previous" || direction == "prev"
        comp_op = "<"
        order_clause = "DESC"
    else
        error("Invalid direction: $direction. Use :next or :previous.")
    end

    #make the SQL query global TODO:
    if full_row
        query = """
            SELECT id_text, embedding_blob
            FROM $MAIN_TABLE_NAME
            WHERE id_text $comp_op ?
            ORDER BY id_text $order_clause
            LIMIT 1;
        """
    else
        query = """
            SELECT id_text
            FROM $MAIN_TABLE_NAME
            WHERE id_text $comp_op ?
            ORDER BY id_text $order_clause
            LIMIT 1;
        """
    end

    rows = SQLite.transaction(db) do
        collect(Tables.namedtupleiterator(DBInterface.execute(db, query, (string(current_id),))))
    end

    if isempty(rows)
        return nothing
    end
    row = rows[1]
    
    if !full_row
        return row.id_text
    else
        #full_row retrieval, obtain the stored data type from the meta table
        meta_rows = SQLite.transaction(db) do
            collect(Tables.namedtupleiterator(DBInterface.execute(db, META_SELECT_ALL_QUERY)))
        end

        if isempty(meta_rows)
            error("Meta table is empty. The meta row should have been initialized during database setup.")
        end
        meta = meta_rows[1]
        T = parse_data_type(meta.data_type)
        embedding_vec = blob_to_embedding(row.embedding_blob, T)
        return (id_text = row.id_text, embedding = embedding_vec, data_type = meta.data_type)
    end
end

function get_adjacent_id(db_path::String, current_id; direction="next", full_row=true)
    db = !isnothing(DB_HANDLE[]) ? DB_HANDLE[] : open_db(db_path)

    try
        result = get_adjacent_id(db, current_id; direction=direction, full_row=full_row)

        return result
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end




"""
Count the number of entries in the main table of the SQLite database.

This function is overloaded to support both an active database connection and a database file path:
- `count_entries(db::SQLite.DB; update_meta::Bool=false)`: Counts entries using an active database connection.
- `count_entries(db_path::String; update_meta::Bool=false)`: Opens the database at the specified path, counts entries, optionally updates the meta table, and then closes the connection.

# Arguments
- For `count_entries(db::SQLite.DB; update_meta::Bool=false)`:
  - `db::SQLite.DB`: An active SQLite database connection.
- For `count_entries(db_path::String; update_meta::Bool=false)`:
  - `db_path::String`: The file path to the SQLite database.
- `update_meta::Bool=false`: When set to `true`, updates the meta table with the current count. If the count is 0, it clears the meta information (i.e., sets `embedding_count` to 0 and resets `embedding_length` if applicable).

# Returns
- The number of entries (an integer) in the main table.
- In the `db_path` overload, returns a `String` error message if an error occurs.

# Example
```julia
# Using an active database connection:
entry_count = count_entries(db, update_meta=true)
println("Number of entries: ", entry_count)

# Using a database file path:
entry_count = count_entries("my_database.db", update_meta=true)
println("Number of entries: ", entry_count)

"""
function count_entries(db::SQLite.DB; update_meta::Bool=false)
    stmt = "SELECT COUNT(*) AS count FROM $MAIN_TABLE_NAME"
    rows = DBInterface.execute(db, stmt) |> DataFrame
    count = rows[1,:].count

    if update_meta
        SQLite.transaction(db) do
            if count > 0
                DBInterface.execute(db, "UPDATE $META_DATA_TABLE_NAME SET embedding_count = ?", (count,))
            else
                # If count is 0, clear the meta information (embedding_count = 0)
                DBInterface.execute(db, "UPDATE $META_DATA_TABLE_NAME SET embedding_count = 0")
            end
        end
    end
    
    return count
end

function count_entries(db_path::String; update_meta::Bool=false)
    db = !isnothing(DB_HANDLE[]) ? DB_HANDLE[] : open_db(db_path)

    try
        count = count_entries(db; update_meta=update_meta)

        return count
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end












end #END MODULE
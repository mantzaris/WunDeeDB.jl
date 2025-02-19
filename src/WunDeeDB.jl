module WunDeeDB

using SQLite
using Tables, DBInterface
using JSON3


export get_supported_data_types, get_supported_endianness_types
        initialize_db, open_db, close_db, delete_db, delete_all_embeddings,
        to_json_embedding, infer_data_type,
        insert_embedding, 
        delete_embedding, 
        update_embedding,
        get_embedding, bulk_get_embedding,
        get_next_id, get_previous_id,
        count_entries, get_embedding_size, random_embeddings

        

###################
#TODO:
#linear exact search for Retrieval (brute force)
#IVF, HSNW
###################

#TODO: add Float8s, https://github.com/JuliaMath/Float8s.jl
const DATA_TYPE_MAP = Dict(
    "Float16"  => Float16,
    "Float32"  => Float32,
    "Float64"  => Float64,
    "BigFloat" => BigFloat,
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

const ENDIANNESS_TYPES = ["small", "big"]

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
           endianness TEXT NOT NULL
       )
       """

const DELETE_EMBEDDINGS_STMT = "DELETE FROM $(MAIN_TABLE_NAME)"

const META_TABLE_FULL_ROW_INSERTION_STMT = """
        INSERT INTO $(META_DATA_TABLE_NAME) (embedding_count, embedding_length, data_type, endianness)
        VALUES (?, ?, ?, ?)
        """

const META_SELECT_ALL_QUERY = "SELECT * FROM $(META_DATA_TABLE_NAME)"
const META_UPDATE_QUERY = "UPDATE $(META_DATA_TABLE_NAME) SET embedding_count = ?"
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
get_supported_endianness_types() -> Vector{String}

Returns a sorted vector of supported endianness as strings.
"""
function get_supported_endianness_types()::Vector{String}
    return sort(collect(keys(ENDIANNESS_TYPES)))
end


#TODO: docstring
#TODO: make endianness optional that it is inferred if not provided
function initialize_db(db_path::String, embedding_length::Int, data_type::String, endianness::String)

    arg_support_string = ""

    if !(data_type in keys(DATA_TYPE_MAP))
        arg_support_string *= "Unsupported data_type, run get_supported_data_types() to get the supported types. "
    end

    if !( endianness in ENDIANNESS_TYPES )
        arg_support_string *= "Unsupported endianness type, run get_supported_endianness_types() to get the supported. "
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

        SQLite.execute(db, CREATE_MAIN_TABLE_STMT)
        SQLite.execute(db, CREATE_META_TABLE_STMT)

        rows = collect(SQLite.Query(db, META_SELECT_ALL_QUERY))

        if isempty(rows) #if empty, insert initial meta information
            #in initial stage set embedding_count to 0 because no embeddings have been added yet
            SQLite.execute(db, META_TABLE_FULL_ROW_INSERTION_STMT, (0, embedding_length, data_type, endianness))
        end

        close_db(db)
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
- `db_path::String`: The file path to the SQLite database. The function will ensure that the directory containing this file exists.

# Returns
- An instance of `SQLite.DB` representing the open and configured database connection.

# Example
```julia
db = open_db("data/mydatabase.sqlite")
# Use the database connection...
SQLite.execute(db, "SELECT * FROM my_table;")
# Don't forget to close the database when done.
close_db(db)
"""
function open_db(db_path::String)
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


""" close_db(db::SQLite.DB)

Close an open SQLite database connection.

This function is a simple wrapper around SQLite.close to ensure that the provided database connection is properly closed when it is no longer needed.

# Arguments

- db::SQLite.DB: The SQLite database connection to be closed.

# Example
```julia
db = open_db("data/mydatabase.sqlite")
# Perform database operations...
close_db(db)
""" 
function close_db(db::SQLite.DB)
    SQLite.close(db)
end


# TODO: doc string (public)
function delete_db(db_path::String)
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

# TODO: doc string (public)
function delete_all_embeddings(db_path::String)
    db = open_db(db_path)
    try
        SQLite.execute(db, DELETE_EMBEDDINGS_STMT) #clear all embeddings
        SQLite.execute(db, META_RESET_STMT) #reset the embedding count to 0
        close_db(db)
        return true
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end


# TODO: doc string (public)
function to_json_embedding(vec::AbstractVector{<:Number})
    return JSON3.write(vec)
end

# TODO: doc string (public)
function infer_data_type(embedding::AbstractVector{<:Number})
    return string(eltype(embedding))
end

#helper to convert an embedding vector to a binary blob:
function embedding_to_blob(embedding::AbstractVector{<:Number})
    return Vector{UInt8}(reinterpret(UInt8, embedding))
end


# doc string private
function update_meta(db::SQLite.DB, count::Int=1)
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, META_SELECT_ALL_QUERY)))
    
    if isempty(rows)
        error("Meta data table is empty. The meta row should have been initialized during database setup.")
    else
        row = rows[1]
        current_count = row.embedding_count
        new_count = current_count + count
        
        SQLite.execute(db, META_UPDATE_QUERY, (new_count,))
    end
end






# TODO: public doc string
function insert_embedding(db::SQLite.DB, collection_name::String, id_input, embedding_input)
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

    #see all embeddings have the same length.
    emb_length = length(embeddings[1])
    for e in embeddings
        if length(e) != emb_length
            error("All embeddings must have the same length")
        end
    end

    local inferred_data_type = infer_data_type(embeddings[1]) #see data type from the first embedding

    #retrieve meta table information using DBInterface.execute with Tables.namedtupleiterator
    meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, META_SELECT_ALL_QUERY)))
    
    if isempty(meta_rows)
        error("Meta table is empty. The meta row should have been initialized during database setup.")
    end
    
    meta = meta_rows[1]
    
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
            SQLite.execute(db, INSERT_EMBEDDING_STMT, p)
        end
        # Update the meta table by incrementing embedding_count by n.
        update_meta(db, count=n)
    end

    return true
end

# TODO: public doc string
function insert_embedding(db_path::String, collection_name::String, id_input, embedding_input)
    db = open_db(db_path)
    try 
        msg = insert_embedding(db, collection_name, id_input, embedding_input)
        close_db(db)
        return msg
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end




function delete_embedding(db::SQLite.DB, id_input)
    # Normalize id_input: if a single ID is passed, wrap it in a one-element array.
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
        SQLite.execute(db, stmt, params)
        
        #update the meta table for bulk deletion, this function should subtract 'n' from the meta embedding_count
        update_meta(db, -n)
    end

    return true
end

# Convenience wrapper: opens/closes the database given a file path.
function delete_embedding(db_path::String, id_input)
    db = open_db(db_path)
    try
        result = delete_embedding(db, id_input)
        close_db(db)
        return result
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end




# TODO: public doc string
function update_embedding(db::SQLite.DB, collection_name::String, id_input, new_embedding_input)
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

    # Retrieve meta table information using the Tables interface.
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
        found_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, check_sql, (string(ids[1]),))))
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

#TODO: public doc string
function update_embedding(db_path::String, id_input, new_embedding_input)
    db = open_db(db_path)
    try
        result = update_embedding(db, id_input, new_embedding_input)
        close_db(db)
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


"""
get_embedding(db::SQLite.DB, collection_name::String, id_text) -> Union{Vector{T}, Nothing} where T

Retrieve the embedding vector for a given identifier from a specified collection (table) in an SQLite database.

This function queries the table `collection_name` for the row where the `id_text` matches the provided identifier (converted to a string). The table is expected to have two columns:
- `embedding_json`: A JSON-encoded string representing the embedding vector.
- `data_type`: A string that specifies the type of the elements in the embedding vector.

The JSON string is parsed into a Julia vector using `JSON3.read` with the element type determined by the helper function `parse_data_type`. If no matching row is found, the function returns `nothing`.

# Arguments
- `db::SQLite.DB`: An active connection to an SQLite database.
- `collection_name::String`: The name of the table (collection) to query.
- `id_text`: The identifier for which the embedding is requested (this value is converted to a string).

# Returns
- A vector representing the parsed embedding if a matching row is found.
- `nothing` if no row with the specified `id_text` exists.

# Example
```julia
embedding = get_embedding(db, "embeddings", "id123")
if embedding === nothing
    println("No embedding found for the given id.")
else
    println("Retrieved embedding: ", embedding)
end
"""
function get_embedding(db::SQLite.DB, collection_name::String, id_text)
    sql = """
    SELECT embedding_json, data_type
    FROM $(collection_name)
    WHERE id_text = ?
    """
    #use DBInterface.execute and convert the results into NamedTuples
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, sql, (string(id_text),))))
    if isempty(rows)
        return nothing
    end

    row = rows[1]
    raw_json = row.embedding_json
    dt_string = row.data_type

    T = parse_data_type(dt_string)
    embedding_vec = JSON3.read(raw_json, Vector{T})

    return embedding_vec
end

""" 
get_embedding(db_path::String, collection_name::String, id_text) -> Union{Vector{T}, Nothing, String} where T

A convenience wrapper for retrieving an embedding vector from a specified SQLite database by using the database file path.

This function opens the SQLite database located at db_path, delegates the retrieval of the embedding to get_embedding(db::SQLite.DB, collection_name, id_text), and then ensures that the database connection is closed. In the event of an error during the operation, the function returns a descriptive error message as a String.

# Arguments
- db_path::String: The file path to the SQLite database.
- collection_name::String: The name of the table (collection) to query.
- id_text: The identifier for which the embedding is requested (converted to a string).

# Returns
- On success: A vector representing the parsed embedding if found, or nothing if no matching row exists.
- On error: A String containing an error message.

Example
```julia
embedding = get_embedding("mydatabase.sqlite", "embeddings", "id123")
if isa(embedding, String)
    println("Error: ", embedding)
elseif embedding === nothing
    println("No embedding found for the given id.")
else
    println("Retrieved embedding: ", embedding)
end
"""
function get_embedding(db_path::String, collection_name::String, id_text)
    db = open_db(db_path)
    try
        vec = get_embedding(db, collection_name, id_text)
        close_db(db)
        return vec
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end


"""
bulk_get_embedding(db::SQLite.DB, collection_name::String, id_texts::Vector{<:AbstractString}) -> Dict{String,Any}

Retrieve embeddings in bulk from the specified collection (table) in an SQLite database, based on a vector of identifier strings.

This function first checks that the number of identifiers does not exceed a predefined bulk limit (`BULK_LIMIT`). It then constructs an SQL query using parameterized placeholders to select rows from the table `collection_name` where the `id_text` is one of the provided identifiers. For each returned row, it:
  - Retrieves the `id_text`, JSON-encoded embedding (`embedding_json`), and `data_type`.
  - Uses the helper function `parse_data_type` to determine the correct type for the embedding.
  - Parses the JSON string into a vector using `JSON3.read`.
  - Inserts the resulting embedding vector into a dictionary with the corresponding `id_text` as the key.

# Arguments
- `db::SQLite.DB`: An active SQLite database connection.
- `collection_name::String`: The name of the table (collection) to query.
- `id_texts::Vector{<:AbstractString}`: A vector of identifier strings for which embeddings should be fetched.

# Returns
- A `Dict{String,Any}` mapping each `id_text` (as a `String`) to its corresponding embedding vector.

# Raises
- Throws an error if the number of identifiers exceeds the limit specified by `BULK_LIMIT`.

# Example
```julia
db = SQLite.DB("mydatabase.sqlite")
ids = ["id1", "id2", "id3"]
embeddings = bulk_get_embedding(db, "embeddings", ids)
for (id, emb) in embeddings
    println("ID: ", id, " -> Embedding: ", emb)
end
"""
function bulk_get_embedding(db::SQLite.DB, collection_name::String, id_texts::Vector{<:AbstractString})
    n = length(id_texts)
    if n > BULK_LIMIT
        error("Bulk get limit exceeded: $n > $BULK_LIMIT")
    end
    placeholders = join(fill("?", n), ", ")
    stmt = """
    SELECT id_text, embedding_json, data_type FROM $(collection_name)
    WHERE id_text IN ($placeholders)
    """
    params = Tuple(string.(id_texts))
    #use DBInterface.execute and convert the result into NamedTuples
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, stmt, params)))
    result = Dict{String,Any}()
    for row in rows
        id = row.id_text
        raw_json = row.embedding_json
        dt_string = row.data_type
        T = parse_data_type(dt_string)
        embedding_vec = JSON3.read(raw_json, Vector{T})
        result[string(id)] = embedding_vec
    end
    return result
end

""" 
bulk_get_embedding(db_path::String, collection_name::String, id_texts::Vector{<:AbstractString}) -> Union{Dict{String,Any}, String}

A convenience wrapper for bulk fetching embeddings from an SQLite database by specifying the database file path.

This function opens the SQLite database located at db_path and calls the primary bulk_get_embedding function to retrieve embeddings for the provided id_texts. The function ensures that the database connection is closed after the operation, even if an error occurs. In case of an error during execution, a descriptive error message is returned as a String.

# Arguments

- db_path::String: The file path to the SQLite database.
- collection_name::String: The name of the table (collection) to query.
- id_texts::Vector{<:AbstractString}: A vector of identifier strings for which embeddings should be fetched.

# Returns

- On success: A Dict{String,Any} mapping each id_text to its corresponding embedding vector.
- On error: A String containing an error message.

# Example
```julia
ids = ["id1", "id2", "id3"]
result = bulk_get_embedding("mydatabase.sqlite", "embeddings", ids)
if isa(result, String)
    println("Error occurred: ", result)
else
    for (id, emb) in result
        println("ID: ", id, " -> Embedding: ", emb)
    end
end
"""
function bulk_get_embedding(db_path::String, collection_name::String, id_texts::Vector{<:AbstractString})
    db = open_db(db_path)
    try
        result = bulk_get_embedding(db, collection_name, id_texts)
        close_db(db)
        return result
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end


#utility functions

"""
get_next_id(db::SQLite.DB, collection_name::String, current_id; full_row::Bool=false) -> Union{Nothing, Any}

Retrieve the entry that immediately follows a given `current_id` in lexicographical (alphabetical) order from the specified collection (table) in an SQLite database.

This function searches for the row in `collection_name` where the `id_text` is lexicographically greater than `current_id`. The SQL query uses the condition `WHERE id_text > ?` and orders the results in ascending lexicographical order (`ORDER BY id_text ASC`), ensuring that the smallest `id_text` greater than `current_id` is returned.

- **When `full_row` is `false` (default):**  
  Only the `id_text` of the next entry is returned.
  
- **When `full_row` is `true`:**  
  The function returns a NamedTuple containing:
  - `id_text`: The identifier of the entry.
  - `embedding`: The embedding vector parsed from the JSON string found in the `embedding_json` field.
  - `data_type`: The data type indicator used for parsing the embedding (with the appropriate type determined by `parse_data_type`).

If no such entry is found (i.e., there is no row with an `id_text` lexicographically greater than `current_id`), the function returns `nothing`.

# Arguments
- `db::SQLite.DB`: An active connection to an SQLite database.
- `collection_name::String`: The name of the table (collection) to query.
- `current_id`: The current identifier for comparison (expected to be a string or a type compatible with lexicographical ordering).
- `full_row::Bool=false`: Optional flag. Set to `true` to retrieve the full row (with parsed embedding); otherwise, only the `id_text` is returned.

# Returns
- If a next entry is found:
  - **Default (`full_row == false`):** Returns the `id_text` of the next entry.
  - **If `full_row == true`:** Returns a NamedTuple with fields `id_text`, `embedding`, and `data_type`.
- If no entry is found: Returns `nothing`.

# Example
```julia
# Retrieve only the next id based on lexicographical ordering:
next_id = get_next_id(db, "embeddings", current_id)
println("Next ID (lexicographical): ", next_id)

# Retrieve the full next row with parsed embedding:
next_row = get_next_id(db, "embeddings", current_id; full_row=true)
if next_row !== nothing
    @show next_row.id_text, next_row.embedding, next_row.data_type
else
    println("No next entry found based on lexicographical ordering.")
end
"""
function get_next_id(db::SQLite.DB, collection_name::String, current_id; full_row::Bool=false)
    if full_row
        query = """
            SELECT id_text, embedding_json, data_type
            FROM $(collection_name)
            WHERE id_text > ?
            ORDER BY id_text ASC
            LIMIT 1;
        """
    else
        query = """
            SELECT id_text
            FROM $(collection_name)
            WHERE id_text > ?
            ORDER BY id_text ASC
            LIMIT 1;
        """
    end
    #use DBInterface.execute with Tables.namedtupleiterator for proper row conversion
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, query, (current_id,))))
    if isempty(rows)
        return nothing
    end
    row = rows[1]
    if !full_row
        return row.id_text
    else
        #parse the JSON embedding using the stored data type
        T = parse_data_type(row.data_type)
        embedding_vec = JSON3.read(row.embedding_json, Vector{T})
        return (id_text = row.id_text, embedding = embedding_vec, data_type = row.data_type)
    end
end

""" 
get_next_id(db_path::String, collection_name::String, current_id; full_row::Bool=false) -> Union{Nothing, Any, String}

A convenience wrapper for retrieving the entry that immediately follows a given current_id (based on lexicographical ordering) from a specified collection using a database file path.

This function opens an SQLite database using the provided db_path and delegates the retrieval of the next entry to get_next_id(db::SQLite.DB, collection_name, current_id; full_row::Bool=false). The lexicographical comparison (WHERE id_text > ?) ensures that the smallest id_text greater than current_id is returned. The database connection is closed after the operation, and any errors encountered are returned as a descriptive string.
Arguments

- db_path::String: The file path to the SQLite database.
- collection_name::String: The name of the table (collection) to query.
- current_id: The current identifier used for comparison.
- full_row::Bool=false: Optional flag. Set to true to retrieve the full row (with parsed embedding); otherwise, only the id_text is returned.

# Returns
- On success:
  - If a next entry is found:
    - Default (full_row == false): Returns the id_text of the next entry.
    - If full_row == true): Returns a NamedTuple with fields id_text, embedding, and data_type.
  - If no entry is found: Returns nothing.
- On error: Returns a String containing the error message.

# Example
```julia
result = get_next_id("mydatabase.sqlite", "embeddings", current_id; full_row=true)
if isa(result, String)
    println("Error: ", result)
elseif result === nothing
    println("No next entry found based on lexicographical ordering.")
else
    @show result.id_text, result.embedding, result.data_type
end
"""
function get_next_id(db_path::String, collection_name::String, current_id; full_row::Bool=false)
    db = open_db(db_path)
    try
        result = get_next_id(db, collection_name, current_id; full_row=full_row)
        close_db(db)
        return result
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end


"""
get_previous_id(db::SQLite.DB, collection_name::String, current_id; full_row::Bool=false) -> Union{Nothing, Any}

Retrieve the entry that immediately precedes a given `current_id` in lexicographical (alphabetical) order from the specified collection (table) in an SQLite database.

This function searches for the row in `collection_name` where the `id_text` is lexicographically less than `current_id`. The comparison is performed using standard string ordering (i.e., alphabetical order), so it is assumed that the `id_text` values are stored as strings. The query orders the results in descending lexicographical order by `id_text`, ensuring that the row with the closest preceding `id_text` is selected.

- **When `full_row` is `false` (default):**  
  The function returns only the `id_text` of the previous entry.

- **When `full_row` is `true`:**  
  The function returns a NamedTuple containing:
  - `id_text`: The identifier of the entry.
  - `embedding`: The embedding vector parsed from the JSON string in the `embedding_json` field.
  - `data_type`: The data type indicator used to parse the embedding (with the type determined by `parse_data_type`).

If no entry is found (i.e., there is no row with an `id_text` lexicographically less than `current_id`), the function returns `nothing`.

# Arguments
- `db::SQLite.DB`: An active connection to an SQLite database.
- `collection_name::String`: The name of the table (collection) to query.
- `current_id`: The current identifier used for comparison (expected to be a string or compatible type).
- `full_row::Bool=false`: Optional flag. Set to `true` to retrieve the full row (with parsed embedding), or `false` to retrieve only the `id_text`.

# Returns
- If a previous entry is found:
  - **Default (`full_row == false`):** Returns the `id_text` of the previous entry.
  - **If `full_row == true`:** Returns a NamedTuple with fields `id_text`, `embedding`, and `data_type`.
- If no previous entry is found: Returns `nothing`.

# Example
```julia
# Retrieve only the previous id based on lexicographical ordering:
prev_id = get_previous_id(db, "embeddings", current_id)
println("Previous ID (lexicographical): ", prev_id)

# Retrieve the full previous row with parsed embedding:
prev_row = get_previous_id(db, "embeddings", current_id; full_row=true)
if prev_row !== nothing
    @show prev_row.id_text, prev_row.embedding, prev_row.data_type
else
    println("No previous entry found based on lexicographical ordering.")
end
"""
function get_previous_id(db::SQLite.DB, collection_name::String, current_id; full_row::Bool=false)
    if full_row
        query = """
            SELECT id_text, embedding_json, data_type
            FROM $(collection_name)
            WHERE id_text < ?
            ORDER BY id_text DESC
            LIMIT 1;
        """
    else
        query = """
            SELECT id_text
            FROM $(collection_name)
            WHERE id_text < ?
            ORDER BY id_text DESC
            LIMIT 1;
        """
    end
    # using DBInterface.execute with Tables.namedtupleiterator to get NamedTuples.
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, query, (current_id,))))
    if isempty(rows)
        return nothing
    end
    row = rows[1]
    if !full_row
        return row.id_text
    else
        T = parse_data_type(row.data_type)
        embedding_vec = JSON3.read(row.embedding_json, Vector{T})
        return (id_text = row.id_text, embedding = embedding_vec, data_type = row.data_type)
    end
end

""" 
get_previous_id(db_path::String, collection_name::String, current_id; full_row::Bool=false) -> Union{Nothing, Any, String}

A convenience wrapper for retrieving the entry immediately preceding a given current_id (based on lexicographical ordering) from a collection by specifying the database file path.

This function opens an SQLite database using the provided db_path and delegates the task to get_previous_id(db::SQLite.DB, collection_name, current_id; full_row::Bool=false). The comparison of id_text values is performed lexicographically (alphabetically), ensuring that the row with the closest preceding identifier is selected. The database connection is properly closed after the operation. In the event of an error, the function returns a descriptive error message as a String.

# Arguments

- db_path::String: The file path to the SQLite database.
- collection_name::String: The name of the table (collection) to query.
- current_id: The current identifier used for comparison (expected to be a string or a compatible type).
- full_row::Bool=false: Optional flag. Set to true to retrieve the full row (with parsed embedding), or false to retrieve only the id_text.

# Returns

- On success:
  - If a previous entry is found:
    - Default (full_row == false): Returns the id_text of the previous entry.
    - If full_row == true: Returns a NamedTuple with fields id_text, embedding, and data_type.
  - If no previous entry is found: Returns nothing.
- On error: Returns a String containing the error message.

# Example
```julia
result = get_previous_id("mydatabase.sqlite", "embeddings", current_id; full_row=true)
if isa(result, String)
    println("Error: ", result)
elseif result === nothing
    println("No previous entry found based on lexicographical ordering.")
else
    @show result.id_text, result.embedding, result.data_type
end
"""
function get_previous_id(db_path::String, collection_name::String, current_id; full_row::Bool=false)
    db = open_db(db_path)
    try
        result = get_previous_id(db, collection_name, current_id; full_row=full_row)
        close_db(db)
        return result
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end



"""
count_entries(db::SQLite.DB, collection_name::String; update_meta::Bool=false) -> Int

Count the total number of entries (rows) in the specified collection (table) from an SQLite database,
and optionally update the associated metadata table.

This function performs an SQL `COUNT(*)` query on the table named `collection_name` to obtain the number of rows.
It retrieves the count from a result set containing a named tuple with the field `count`. Optionally, if the keyword
argument `update_meta` is set to `true`, the function updates the metadata table, which is assumed to be named
`\$(collection_name)_meta`. The metadata update behavior is as follows:
- **If count > 0:** Update the `row_num` field with the current count.
- **If count == 0:** Clear the metadata by setting `row_num` to 0 and `vector_length` to `NULL`.

# Arguments
- `db::SQLite.DB`: An open connection to an SQLite database.
- `collection_name::String`: The name of the table whose entries are to be counted.
- `update_meta::Bool=false`: Optional flag indicating whether to update the metadata table based on the count.

# Returns
- `Int`: The number of entries (rows) in the specified table.

# Example
```julia
db = SQLite.DB("mydatabase.sqlite")
num_entries = count_entries(db, "embeddings"; update_meta=true)
println("Number of entries: ", num_entries)
"""
function count_entries(db::SQLite.DB, collection_name::String; update_meta::Bool=false)
    stmt = "SELECT COUNT(*) AS count FROM $(collection_name)"
    #using DBInterface.execute with Tables.namedtupleiterator so that we get a NamedTuple with field "count"
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, stmt)))
    count = rows[1].count
    
    if update_meta
        if count > 0
            update_stmt = "UPDATE $(collection_name)_meta SET row_num = ?"
            DBInterface.execute(db, update_stmt, (count,))
        else
            #if count is 0, clear the meta information.
            update_stmt = "UPDATE $(collection_name)_meta SET row_num = 0, vector_length = NULL"
            DBInterface.execute(db, update_stmt)
        end
    end
    
    return count
end

""" 
count_entries(db_path::String, collection_name::String; update_meta::Bool=false) -> Union{Int, String}

A convenience wrapper for counting entries in a specified table of an SQLite database by using a file path. This function opens an SQLite database using db_path, delegates the counting to count_entries(db::SQLite.DB, collection_name; update_meta::Bool=false), and then ensures that the database connection is closed after the operation. In case an error occurs during the process, the function returns a descriptive error message as a string.

# Arguments

- db_path::String: The file path to the SQLite database.
- collection_name::String: The name of the table whose entries are to be counted.
- update_meta::Bool=false: Optional flag indicating whether to update the metadata table.

# Returns

- On success: An Int representing the number of entries in the table.
- On error: A String containing the error message.

# Example

```julia
result = count_entries("mydatabase.sqlite", "embeddings"; update_meta=true)
if isa(result, String)
    println("Error occurred: ", result)
else
    println("Number of entries: ", result)
end
"""
function count_entries(db_path::String, collection_name::String; update_meta::Bool=false)
    db = open_db(db_path)
    try
        count = count_entries(db, collection_name; update_meta=update_meta)
        close_db(db)
        return count
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end




"""
get_embedding_size(db::SQLite.DB, collection_name::String) -> Int

Fetch the embedding vector size for a given collection from the metadata table in an SQLite database.

This function queries the metadata table associated with the collection, which is assumed to be named 
`\$(collection_name)_meta`, and retrieves the value of the `vector_length` column. If no metadata is found, 
the function returns `0`; otherwise, it returns the embedding size from the first row of the query result.

# Arguments
- `db::SQLite.DB`: An active connection to an SQLite database.
- `collection_name::String`: The base name of the collection whose embedding size is to be retrieved. The
  metadata table is expected to be named as `\$(collection_name)_meta`.

# Returns
- `Int`: The embedding vector size if metadata is found, or `0` if no metadata exists.

# Example
```julia
db = SQLite.DB("mydatabase.sqlite")
embedding_size = get_embedding_size(db, "embeddings")
println("Embedding size: ", embedding_size)
"""
function get_embedding_size(db::SQLite.DB, collection_name::String)
    stmt = "SELECT vector_length FROM $(collection_name)_meta"
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, stmt)))
    if isempty(rows)
        return 0
    else
        return rows[1].vector_length
    end
end

""" 
get_embedding_size(db_path::String, collection_name::String) -> Union{Int, String}

A convenience wrapper for retrieving the embedding size from an SQLite database using a file path.

This function opens an SQLite database using the provided db_path and delegates the retrieval of the embedding size to get_embedding_size(db::SQLite.DB, collection_name). It ensures that the database connection is closed after the operation. If an error occurs during the process, the function returns a string with the error message.

# Arguments

- db_path::String: The file path to the SQLite database.
- collection_name::String: The base name of the collection. The metadata table is expected to be named \$(collection_name)_meta.

# Returns
- On success: An Int representing the embedding vector size.
- On error: A String containing an error message.

# Example
```julia
embedding_size = get_embedding_size("mydatabase.sqlite", "embeddings")
if isa(embedding_size, String)
    println("Error occurred: ", embedding_size)
else
    println("Embedding size: ", embedding_size)
end
""" 
function get_embedding_size(db_path::String, collection_name::String)
    db = open_db(db_path)
    try
        size = get_embedding_size(db, collection_name)
        close_db(db)
        return size
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end



"""
random_embeddings(db::SQLite.DB, collection_name::String, num::Int) -> Vector{Dict{String,Any}}

Retrieve a specified number of random embedding records from a given collection in an SQLite database.

This function performs an SQL query against the specified `collection_name` (which is typically the name
of a table in the database) to fetch up to `num` random rows. Each row is expected to have the following columns:
- `id_text`: An identifier for the embedding.
- `embedding_json`: A JSON string representing the embedding vector.
- `data_type`: A string indicating the data type of the elements within the embedding.

The JSON in `embedding_json` is parsed using JSON3 into a `Vector{T}`, where the type `T` is determined by
the `data_type` field (via the helper function `parse_data_type`).

# Arguments
- `db::SQLite.DB`: An active connection to an SQLite database.
- `collection_name::String`: The name of the table/collection from which to retrieve embeddings.
- `num::Int`: The number of random embeddings to retrieve. This value must be between 1 and `BULK_LIMIT`.

# Returns
A `Vector{Dict{String, Any}}` where each dictionary has the following keys:
- `"id_text"`: The unique identifier for the embedding.
- `"embedding"`: The embedding vector (parsed from JSON).
- `"data_type"`: The data type string that was used to parse the embedding.

# Throws
- Raises an error if `num` is less than 1 or greater than `BULK_LIMIT`.

# Example
```julia
db = SQLite.DB("mydatabase.sqlite")
embeddings = random_embeddings(db, "embeddings_table", 5)
for record in embeddings
    @show record["id_text"], record["embedding"], record["data_type"]
end

"""
function random_embeddings(db::SQLite.DB, collection_name::String, num::Int)
    if num < 1 || num > BULK_LIMIT
        error("Requested number of random embeddings must be between 1 and $BULK_LIMIT")
    end

    stmt = """
        SELECT id_text, embedding_json, data_type 
        FROM $(collection_name)
        ORDER BY RANDOM() 
        LIMIT ?;
    """
    # DBInterface.execute together with Tables.namedtupleiterator.
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, stmt, (num,))))
    
    results = Vector{Dict{String,Any}}(undef, length(rows))
    for (i, row) in enumerate(rows)
        T = parse_data_type(row.data_type)
        embedding = JSON3.read(row.embedding_json, Vector{T})
        results[i] = Dict("id_text" => row.id_text, "embedding" => embedding, "data_type" => row.data_type)
    end

    return results
end

""" 
random_embeddings(db_path::String, collection_name::String, num::Int) -> Union{Vector{Dict{String,Any}}, String}

A convenience wrapper to retrieve random embedding records by providing a database file path.

This function opens an SQLite database using the file path db_path, then delegates the task of fetching random embeddings to the random_embeddings(db::SQLite.DB, collection_name, num) function. Once the data has been retrieved (or an error occurs), the database connection is properly closed. In the event of an exception during retrieval, the function returns a string that describes the error.

# Arguments
- db_path::String: The file path to the SQLite database.
- collection_name::String: The name of the table/collection from which to retrieve embeddings.
- num::Int: The number of random embeddings to retrieve. This value must be between 1 and BULK_LIMIT.

# Returns

    On success: A Vector{Dict{String, Any}} where each dictionary contains keys "id_text", "embedding", and "data_type".
    On failure: A String containing an error message.

# Example

result = random_embeddings("mydatabase.sqlite", "embeddings_table", 5)
if typeof(result) == String
    println("An error occurred: ", result)
else
    for record in result
        @show record["id_text"], record["embedding"], record["data_type"]
    end
end
"""
function random_embeddings(db_path::String, collection_name::String, num::Int)
    db = open_db(db_path)
    try
        result = random_embeddings(db, collection_name, num)
        close_db(db)
        return result
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end





end #END MODULE
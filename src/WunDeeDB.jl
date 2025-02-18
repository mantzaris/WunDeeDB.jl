module WunDeeDB

using SQLite
using Tables, DBInterface
using JSON3


export get_supported_data_types, get_supported_endianness_types
        initialize_db, open_db, close_db, delete_db, delete_all_embeddings,
        to_json_embedding, infer_data_type,
        insert_embedding, bulk_insert_embedding, 
        delete_embedding, bulk_delete_embedding, 
        update_embedding, bulk_update_embedding,
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

const main_table_name = "Embeddings"
const meta_data_table_name = "EmbeddingsMetaData"

const create_main_table_stmt = """
        CREATE TABLE IF NOT EXISTS $(main_table_name) (
            id_text TEXT PRIMARY KEY,
            embedding_blob BLOB NOT NULL
        )
        """

const create_meta_table_stmt = """
       CREATE TABLE IF NOT EXISTS $(meta_data_table_name) (
	   embedding_count BIGINT,
	   embedding_length INT,
	   data_type TEXT NOT NULL,
           endianness TEXT NOT NULL
       )
       """

const delete_embeddings_stmt = "DELETE FROM $(main_table_name)"

const meta_table_full_row_insertion_stmt = """
        INSERT INTO $(meta_data_table_name) (embedding_count, embedding_length, data_type, endianness)
        VALUES (?, ?, ?, ?)
        """

const meta_select_all_query = "SELECT * FROM $(meta_data_table_name)"
const meta_update_query = "UPDATE $(meta_data_table_name) SET embedding_count = ?"
const meta_reset_stmt = "UPDATE $(meta_data_table_name) SET embedding_count = 0"


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

        SQLite.execute(db, create_main_table_stmt)
        SQLite.execute(db, create_meta_table_stmt)

        rows = collect(SQLite.Query(db, meta_select_all_query))

        if isempty(rows) #if empty, insert initial meta information
            #in initial stage set embedding_count to 0 because no embeddings have been added yet
            SQLite.execute(db, meta_table_full_row_insertion_stmt, (0, embedding_length, data_type, endianness))
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
        SQLite.execute(db, delete_embeddings_stmt) #clear all embeddings
        SQLite.execute(db, meta_reset_stmt) #reset the embedding count to 0
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
    elty = eltype(embedding)
    return string(elty) 
end


# doc string private
function update_meta(db::SQLite.DB, count::Int=1)
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, meta_select_all_query)))
    
    if isempty(rows)
        error("Meta data table is empty. The meta row should have been initialized during database setup.")
    else
        row = rows[1]
        current_count = row.embedding_count
        new_count = current_count + count
        
        SQLite.execute(db, meta_update_query, (new_count,))
    end
end




















#DELETE

function update_meta_delete(db::SQLite.DB, meta_table::String)
    q_str = "SELECT row_num, vector_length FROM $(meta_table)"
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, q_str))) #table interface!!! 
    if isempty(rows)
        return
    end

    row = rows[1]
    old_row_num = row.row_num

    old_vec_len = row.vector_length

    new_row_num = max(old_row_num - 1, 0)

    if new_row_num == 0
        update_str = "UPDATE $(meta_table) SET row_num = 0, vector_length = NULL"
        SQLite.execute(db, update_str)
    else
        update_str = "UPDATE $(meta_table) SET row_num = ?"
        SQLite.execute(db, update_str, (new_row_num,))
    end
end

# Bulk deletion meta update: subtract count from row_num.
function bulk_update_meta_delete(db::SQLite.DB, meta_table::String, count::Int)
    q_str = "SELECT row_num, vector_length FROM $(meta_table)"
    # Use DBInterface.execute combined with Tables.namedtupleiterator
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, q_str)))
    if isempty(rows)
        return
    end
    row = rows[1]
    old_row_num = row.row_num
    new_row_num = max(old_row_num - count, 0)
    if new_row_num == 0
        stmt = "UPDATE $(meta_table) SET row_num = 0, vector_length = NULL"
        DBInterface.execute(db, stmt)
    else
        stmt = "UPDATE $(meta_table) SET row_num = ?"
        DBInterface.execute(db, stmt, (new_row_num,))
    end
end


"""
insert_embedding(db::SQLite.DB, collection_name::String, id_text, embedding::AbstractVector{<:Number}; 
                     data_type::Union{Nothing,String}=nothing) -> String

Insert a single embedding record into the specified collection (table) within an SQLite database.

This function performs the following steps:
1. **Data Type Handling:**  
   If `data_type` is not provided, it is inferred from the embedding using `infer_data_type(embedding)`.
   
2. **Embedding Conversion:**  
   The embedding vector is converted to a JSON string using `to_json_embedding(embedding)`.

3. **Atomic Insertion and Metadata Update:**  
   The insertion of the record and the subsequent update of the associated metadata table (named `\$(collection_name)_meta`) are performed within a single SQLite transaction.  
   The metadata is updated via `update_meta(db, "\$(collection_name)_meta", length(embedding))`, which records the dimension of the embedding.

# Arguments
- `db::SQLite.DB`: An active SQLite database connection.
- `collection_name::String`: The name of the table where the embedding record is stored.
- `id_text`: The unique identifier for the record (converted to a string for the query).
- `embedding::AbstractVector{<:Number}`: The embedding vector to be inserted.
- `data_type::Union{Nothing,String}=nothing`: Optional. A string specifying the data type of the embedding values. If not provided, the data type is inferred from the embedding.

# Returns
- A `String` with the value `"true"` indicating a successful insertion.

# Example
```julia
db = SQLite.DB("mydatabase.sqlite")
id = "record123"
emb = [0.1, 0.2, 0.3, 0.4]
result = insert_embedding(db, "embeddings", id, emb)
println("Insert result: ", result)  # Expected output: "true"
"""
function insert_embedding(db::SQLite.DB, collection_name::String, id_text, embedding::AbstractVector{<:Number}; 
    data_type::Union{Nothing,String}=nothing)
    if data_type === nothing
        data_type = infer_data_type(embedding)
    end
    emb_json = to_json_embedding(embedding)
    #put both the insert and meta update in a transaction.
    SQLite.transaction(db) do
        stmt = """
        INSERT INTO $collection_name (id_text, embedding_json, data_type)
        VALUES (?, ?, ?)
        """
        SQLite.execute(db, stmt, (string(id_text), emb_json, data_type))
        update_meta(db, "$(collection_name)_meta", length(embedding))
    end
    return "true"
end

""" 
insert_embedding(db_path::String, collection_name::String, id_text, embedding::AbstractVector{<:Number}; data_type::Union{Nothing,String}=nothing) -> Union{String}

A convenience wrapper for inserting a single embedding record into an SQLite database by specifying the database file path.

This function:
- Opens the SQLite database at the given db_path.
- Delegates the record insertion to insert_embedding(db, collection_name, id_text, embedding; data_type=data_type).
- Ensures that the database connection is closed after the operation.
- Returns a descriptive error message if an exception occurs.

# Arguments
- db_path::String: The file path to the SQLite database.
- collection_name::String: The name of the table where the embedding will be inserted.
- id_text: The unique identifier for the record (converted to a string for the query).
- embedding::AbstractVector{<:Number}: The embedding vector to be inserted.
- data_type::Union{Nothing,String}=nothing: Optional. A string specifying the data type of the embedding values. If omitted, the data type is inferred from the embedding.

# Returns
- On success: A String with the value "true".
- On error: A String containing a descriptive error message.

# Example
```julia
result = insert_embedding("mydatabase.sqlite", "embeddings", "record123", [0.1, 0.2, 0.3, 0.4])
if startswith(result, "Error:")
    println("Insert failed: ", result)
else
    println("Record inserted successfully.")
end
"""
function insert_embedding(db_path::String, collection_name::String, id_text, embedding::AbstractVector{<:Number}; 
    data_type::Union{Nothing,String}=nothing)
    db = open_db(db_path)
    try 
        msg = insert_embedding(db, collection_name, id_text, embedding; data_type=data_type)
        close_db(db)
        return msg
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end


"""
bulk_insert_embedding(db::SQLite.DB, collection_name::String, 
                            id_texts::Vector{<:AbstractString}, embeddings::Vector{<:AbstractVector{<:Number}};
                            data_type::Union{Nothing,String}=nothing) -> String

Bulk insert multiple embedding records into a specified collection (table) in an SQLite database.

This function performs several validation and insertion steps:

1. **Validation:**
   - Checks that the number of identifiers (in `id_texts`) does not exceed a predefined limit (`BULK_LIMIT`).
   - Verifies that the number of embedding vectors matches the number of IDs.
   - Ensures that all embedding vectors have the same length.
   - If `data_type` is not provided, it infers the data type from the first embedding vector using `infer_data_type`.

2. **Parameter Preparation:**
   - Constructs a list of tuples, each containing:
     - The identifier (converted to a string).
     - The JSON-encoded embedding vector (using `JSON3.write`).
     - The determined `data_type`.

3. **Insertion Transaction:**
   - Executes an INSERT statement for each record within a single SQLite transaction.
   - After all records are inserted, updates the associated metadata table (assumed to be named `\$(collection_name)_meta`) via `update_meta_bulk` to reflect the new embedding dimension and row count.

# Arguments
- `db::SQLite.DB`: An active connection to an SQLite database.
- `collection_name::String`: The name of the table where embeddings are stored.
- `id_texts::Vector{<:AbstractString}`: A vector of identifier strings for the new records.
- `embeddings::Vector{<:AbstractVector{<:Number}}`: A vector of embedding vectors to be inserted. All embeddings must be of the same length.
- `data_type::Union{Nothing, String}=nothing`: Optional. A string representing the data type of the embedding values. If omitted, the data type is inferred from the first embedding.

# Returns
- A `String` with the value `"true"` if the bulk insertion is successful.

# Raises
- An error if the number of identifiers exceeds `BULK_LIMIT`.
- An error if the number of embeddings does not match the number of IDs.
- An error if the embedding vectors do not all have the same length.

# Example
```julia
db = SQLite.DB("mydatabase.sqlite")
ids = ["id1", "id2", "id3"]
embs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
result = bulk_insert_embedding(db, "embeddings", ids, embs)
println("Bulk insert result: ", result)
"""
function bulk_insert_embedding(db::SQLite.DB, collection_name::String, 
    id_texts::Vector{<:AbstractString}, embeddings::Vector{<:AbstractVector{<:Number}};
    data_type::Union{Nothing,String}=nothing)

    n = length(id_texts)
    if n > BULK_LIMIT
        error("Bulk insert limit exceeded: $n > $BULK_LIMIT")
    end
    if length(embeddings) != n
        error("Mismatch between number of IDs and embeddings")
    end

    embedding_length = length(embeddings[1])
    for e in embeddings
        if length(e) != embedding_length
            error("All embeddings must have the same length")
        end
    end

    if data_type === nothing
        data_type = infer_data_type(embeddings[1])
    end

    #build parameters for each row
    params = [(string(id_texts[i]), JSON3.write(embeddings[i]), data_type) for i in 1:n]

    SQLite.transaction(db) do
        stmt = """
        INSERT INTO $collection_name (id_text, embedding_json, data_type)
        VALUES (?, ?, ?)
        """
        #TODO improve?..
        #INSERTiING EACH ROW INDEPENDENTLY
        for p in params
            SQLite.execute(db, stmt, p)
        end
        update_meta_bulk(db, "$(collection_name)_meta", embedding_length, n)
    end
    return "true"
end

""" 
bulk_insert_embedding(db_path::String, collection_name::String, id_texts::Vector{<:AbstractString}, embeddings::Vector{<:AbstractVector{<:Number}}; data_type::Union{Nothing,String}=nothing) -> Union{String}

A convenience wrapper for bulk inserting embedding records into an SQLite database using a database file path.

This function opens the SQLite database located at db_path, calls the primary bulk_insert_embedding to insert the specified records into the given collection (table), and ensures that the database connection is closed. If an error occurs during the insertion process, a descriptive error message is returned.

    # Arguments
- db_path::String: The file path to the SQLite database.
- collection_name::String: The name of the table where embeddings will be inserted.
- id_texts::Vector{<:AbstractString}: A vector of identifier strings for the new records.
- embeddings::Vector{<:AbstractVector{<:Number}}: A vector of embedding vectors to be inserted. All embeddings must have the same length.
- data_type::Union{Nothing, String}=nothing: Optional. A string representing the data type of the embedding values. If omitted, it is inferred from the first embedding.

# Returns
- On success: A String with the value "true".
- On error: A String containing a descriptive error message.

# Example
```julia
result = bulk_insert_embedding("mydatabase.sqlite", "embeddings", ["id1", "id2"], [[1.0, 2.0], [3.0, 4.0]])
if startswith(result, "Error:")
    println("Bulk insert failed: ", result)
else
    println("Bulk insert successful: ", result)
end
"""
function bulk_insert_embedding(db_path::String, collection_name::String, 
    id_texts::Vector{<:AbstractString}, embeddings::Vector{<:AbstractVector{<:Number}};
    data_type::Union{Nothing,String}=nothing)
    db = open_db(db_path)
    try
        msg = bulk_insert_embedding(db, collection_name, id_texts, embeddings; data_type=data_type)
        close_db(db)
        return msg
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end


"""
delete_embedding(db::SQLite.DB, collection_name::String, id_text) -> String

Delete a single embedding record from the specified collection (table) in an SQLite database.

This function performs the following steps within a single SQLite transaction:

1. **Record Existence Check:**  
   It verifies that a record with the specified `id_text` exists in the table `collection_name` by executing a SELECT query.  
   If no matching record is found, an error with the message `"notfound"` is raised.

2. **Deletion:**  
   If the record exists, the function deletes the corresponding row from the table using a DELETE SQL command.

3. **Metadata Update:**  
   After deleting the record, it calls `update_meta_delete` on the associated metadata table (assumed to be named `\$(collection_name)_meta`) to update any metadata related to the deletion.

# Arguments
- `db::SQLite.DB`: An active SQLite database connection.
- `collection_name::String`: The name of the table from which the embedding record should be deleted.
- `id_text`: The identifier of the record to delete (this value is converted to a string for SQL operations).

# Returns
- A `String` with the value `"true"` if the deletion (and metadata update) is successful.

# Raises
- An error with the message `"notfound"` if no record with the specified `id_text` exists.

# Example
```julia
db = SQLite.DB("mydatabase.sqlite")
result = delete_embedding(db, "embeddings", "record123")
if result == "true"
    println("Record deleted successfully.")
else
    println("Deletion error: ", result)
end
"""
function delete_embedding(db::SQLite.DB, collection_name::String, id_text)
    SQLite.transaction(db) do
        check_sql = """
        SELECT 1 AS found FROM $(collection_name) WHERE id_text = ?
        """
        #use DBInterface.execute with Tables.namedtupleiterator for consistent row conversion
        found_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, check_sql, (string(id_text),))))
        if isempty(found_rows)
            error("notfound")
        end

        delete_sql = """
        DELETE FROM $(collection_name)
        WHERE id_text = ?
        """
        SQLite.execute(db, delete_sql, (string(id_text),))
        
        update_meta_delete(db, "$(collection_name)_meta")
    end
    return "true"
end

""" 
delete_embedding(db_path::String, collection_name::String, id_text) -> Union{String}

A convenience wrapper for deleting a single embedding record from an SQLite database by specifying the database file path.

This function opens the SQLite database at db_path, calls the primary delete_embedding function to delete the specified record, and then ensures that the database connection is properly closed. If an error occurs during deletion, a descriptive error message is returned as a String.

# Arguments
- db_path::String: The file path to the SQLite database.
- collection_name::String: The name of the table from which the record is to be deleted.
- id_text: The identifier of the record to delete (converted to a string for SQL operations).

# Returns
- On success: A String with the value "true" indicating the record was successfully deleted.
- On error: A String containing a descriptive error message.

# Example
```julia
result = delete_embedding("mydatabase.sqlite", "embeddings", "record123")
if startswith(result, "Error:")
    println("Deletion failed: ", result)
else
    println("Record deleted successfully.")
end
"""
function delete_embedding(db_path::String, collection_name::String, id_text)
    db = open_db(db_path)
    try
        msg = delete_embedding(db, collection_name, id_text)
        close_db(db)
        return msg
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end


"""
bulk_delete_embedding(db::SQLite.DB, collection_name::String, id_texts::Vector{<:AbstractString}) -> String

Delete multiple embedding records from the specified collection (table) in an SQLite database.

This function removes rows from the table named `collection_name` where the `id_text` is one of the provided identifiers.
It enforces a limit on the number of identifiers via the predefined constant `BULK_LIMIT` to prevent overly large transactions.
If the number of identifiers exceeds `BULK_LIMIT`, an error is raised.

The deletion process is executed within an SQLite transaction to ensure atomicity. After the deletion, the function updates the
associated metadata table (assumed to be named `\$(collection_name)_meta`) by calling `bulk_update_meta_delete`, which subtracts the
number of deleted rows from the stored metadata.

# Arguments
- `db::SQLite.DB`: An active connection to an SQLite database.
- `collection_name::String`: The name of the table from which embeddings are to be deleted.
- `id_texts::Vector{<:AbstractString}`: A vector of identifier strings corresponding to the rows to delete.

# Returns
- A `String` with the value `"true"` if the deletion is successful.

# Raises
- An error if the number of identifiers exceeds `BULK_LIMIT`.
- Any SQL errors encountered during the deletion or metadata update will propagate, causing the transaction to roll back.

# Example
```julia
db = SQLite.DB("mydatabase.sqlite")
ids = ["id1", "id2", "id3"]
result = bulk_delete_embedding(db, "embeddings", ids)
println(result)  # Should print "true" if the deletion and metadata update are successful.

"""
function bulk_delete_embedding(db::SQLite.DB, collection_name::String, id_texts::Vector{<:AbstractString})
    n = length(id_texts)
    if n > BULK_LIMIT
        error("Bulk delete limit exceeded: $n > $BULK_LIMIT")
    end
    SQLite.transaction(db) do
        #build a string of comma‐separated placeholders
        placeholders = join(fill("?", n), ", ")
        stmt = "DELETE FROM $collection_name WHERE id_text IN ($placeholders)"
        #convert each id_text to a string and pack them into a tuple
        params = Tuple(string.(id_texts))
        SQLite.execute(db, stmt, params)
        
        #update the meta table to subtract the deleted rows
        bulk_update_meta_delete(db, "$(collection_name)_meta", n)
    end
    return "true"
end

""" 
bulk_delete_embedding(db_path::String, collection_name::String, id_texts::Vector{<:AbstractString}) -> Union{String}

A convenience wrapper for deleting multiple embedding records from an SQLite database using a database file path.

This function opens the SQLite database located at db_path, calls the primary bulk_delete_embedding function to remove the specified rows from the given collection (table), and then ensures that the database connection is properly closed. If an error occurs during the deletion or metadata update, a descriptive error message is returned as a String.

# Arguments
- db_path::String: The file path to the SQLite database.
- collection_name::String: The name of the table from which embeddings are to be deleted.
- id_texts::Vector{<:AbstractString}: A vector of identifier strings corresponding to the rows to delete.

# Returns
- On success: A String with the value "true" indicating the deletion was successful.
- On error: A String containing a descriptive error message.

# Example
```juila
result = bulk_delete_embedding("mydatabase.sqlite", "embeddings", ["id1", "id2", "id3"])
if startswith(result, "Error:")
    println("Bulk delete failed: ", result)
else
    println("Bulk delete successful: ", result)
end
"""
function bulk_delete_embedding(db_path::String, collection_name::String, id_texts::Vector{<:AbstractString})
    db = open_db(db_path)
    try
        msg = bulk_delete_embedding(db, collection_name, id_texts)
        close_db(db)
        return msg
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end




#UPDATE

"""
update_embedding(db::SQLite.DB, collection_name::String, id_text, new_embedding::AbstractVector{<:Number};
                     data_type::Union{Nothing,String}=nothing) -> String

Update the embedding vector for a specified record in the given collection (table) within an SQLite database.

This function performs several key steps within a single SQLite transaction:
1. **Record Existence Check:**  
   It first verifies that a record with the given `id_text` exists in the table `collection_name`.  
   If no record is found, the function raises an error with the message `"notfound"`.

2. **Metadata Retrieval and Dimension Validation:**  
   The function retrieves metadata from the associated metadata table (named as `\$(collection_name)_meta`) to obtain the stored embedding vector length (`vector_length`).  
   If the metadata is missing, an error is raised indicating that no metadata was found.  
   It then compares the stored length with the length of the new embedding vector (`new_embedding`).  
   If the lengths do not match, an error is raised with details about the mismatch.

3. **Embedding Conversion and Data Type Determination:**  
   The new embedding vector is converted to a JSON string using `to_json_embedding`.  
   If the optional `data_type` parameter is not provided, it is inferred from the new embedding using `infer_data_type`.

4. **Record Update:**  
   Finally, the function updates the record in the table `collection_name` by setting the `embedding_json` and `data_type` fields for the row where `id_text` matches the provided identifier.

# Arguments
- `db::SQLite.DB`: An active connection to an SQLite database.
- `collection_name::String`: The name of the table containing the embeddings.
- `id_text`: The identifier of the record to update (converted to a string for the query).
- `new_embedding::AbstractVector{<:Number}`: The new embedding vector to be stored.
- `data_type::Union{Nothing, String}=nothing`: Optional. A string representing the data type of the embedding values. If not provided, the data type is inferred from `new_embedding`.

# Returns
- A `String` with the value `"true"` indicating a successful update.

# Raises
- `"notfound"`: If no record with the specified `id_text` exists in `collection_name`.
- An error if no metadata is found in the associated metadata table.
- An error if the new embedding's length does not match the stored `vector_length`.

# Example
```julia
db = SQLite.DB("mydatabase.sqlite")
id = "record123"
new_vec = [0.1, 0.2, 0.3, 0.4]
result = update_embedding(db, "embeddings", id, new_vec)
if result == "true"
    println("Embedding updated successfully.")
else
    println("Update failed: ", result)
end
"""
function update_embedding(db::SQLite.DB, collection_name::String, id_text, new_embedding::AbstractVector{<:Number};
    data_type::Union{Nothing,String}=nothing)
    SQLite.transaction(db) do
        #check that the record exists
        check_sql = """
        SELECT 1 AS found FROM $(collection_name) WHERE id_text = ?
        """
        rows_found = collect(Tables.namedtupleiterator(DBInterface.execute(db, check_sql, (string(id_text),))))
        if isempty(rows_found)
            error("notfound")
        end

        #retrieve metadata to validate the embedding dimension
        meta_table = "$(collection_name)_meta"
        meta_sql = "SELECT vector_length FROM $(meta_table)"
        meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, meta_sql)))
        if isempty(meta_rows)
            error("No metadata found in $(meta_table); can't validate dimension.")
        end

        stored_length = meta_rows[1].vector_length
        new_length = length(new_embedding)
        if stored_length != new_length
            error("Vector length mismatch: stored=$(stored_length), new=$(new_length)")
        end

        #convert the new embedding to JSON
        emb_json = to_json_embedding(new_embedding)
        if data_type === nothing
            data_type = infer_data_type(new_embedding)
        end

        #update the record
        update_sql = """
        UPDATE $(collection_name)
        SET embedding_json = ?, data_type = ?
        WHERE id_text = ?
        """
        DBInterface.execute(db, update_sql, (emb_json, data_type, string(id_text)))
    end
    return "true"
end

""" 
update_embedding(db_path::String, collection_name::String, id_text, new_embedding::AbstractVector{<:Number}; data_type::Union{Nothing,String}=nothing) -> Union{String}

A convenience wrapper for updating an embedding vector using a database file path.

This function opens the SQLite database specified by db_path and then calls the primary update_embedding function to update the embedding for the specified record in the collection (table). It ensures that the database connection is properly closed after the operation. If an error occurs during the update, a descriptive error message is returned as a String.

# Arguments
- db_path::String: The file path to the SQLite database.
- collection_name::String: The name of the table containing the embeddings.
- id_text: The identifier of the record to update.
- new_embedding::AbstractVector{<:Number}: The new embedding vector to be stored.
- data_type::Union{Nothing, String}=nothing: Optional. A string representing the data type of the embedding values. If omitted, it is inferred from new_embedding.

# Returns
- On success: A String with the value "true".
- On error: A String containing a descriptive error message.

# Example
```julia
result = update_embedding("mydatabase.sqlite", "embeddings", "record123", [0.1, 0.2, 0.3, 0.4])
if startswith(result, "Error:")
    println("Update failed: ", result)
else
    println("Embedding updated successfully.")
end
"""
function update_embedding(db_path::String, collection_name::String, id_text, new_embedding::AbstractVector{<:Number};
    data_type::Union{Nothing,String}=nothing)
    db = open_db(db_path)
    try
        msg = update_embedding(db, collection_name, id_text, new_embedding; data_type=data_type)
        close_db(db)
        return msg
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end



"""
bulk_update_embedding(db::SQLite.DB, collection_name::String, 
        id_texts::Vector{<:AbstractString}, new_embeddings::Vector{<:AbstractVector{<:Number}};
        data_type::Union{Nothing,String}=nothing) -> String

Perform a bulk update of embedding vectors in the specified collection (table) within an SQLite database.

This function updates multiple rows at once by setting a new embedding (stored as a JSON string) and updating the
data type for each specified identifier in `id_texts`. The function verifies that:
  - The number of identifiers does not exceed a predefined bulk limit (`BULK_LIMIT`).
  - The number of new embedding vectors matches the number of identifiers.
  - All provided embedding vectors have the same length.

If the optional `data_type` is not provided, it is inferred from the first embedding vector using `infer_data_type`.

The updates are executed within a single SQLite transaction to ensure atomicity. For each identifier, the function:
  - Converts the corresponding embedding vector to a JSON string using `to_json_embedding`.
  - Executes an UPDATE statement to set `embedding_json` and `data_type` for the row where `id_text` matches the identifier.

# Arguments
- `db::SQLite.DB`: An active connection to an SQLite database.
- `collection_name::String`: The name of the table containing the embeddings.
- `id_texts::Vector{<:AbstractString}`: A vector of identifier strings corresponding to the rows to update.
- `new_embeddings::Vector{<:AbstractVector{<:Number}}`: A vector of new embedding vectors. Every embedding must be of the same length.
- `data_type::Union{Nothing, String}=nothing`: Optional. A string representing the data type of the embeddings. If omitted, the data type is inferred from the first embedding.

# Returns
- A `String` ("true") if the update is successful.

# Raises
- An error if:
  - The number of identifiers exceeds `BULK_LIMIT`.
  - There is a mismatch between the number of identifiers and the number of new embeddings.
  - The provided embedding vectors do not all have the same length.

# Example
```julia
db = SQLite.DB("mydatabase.sqlite")
ids = ["id1", "id2", "id3"]
new_embs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
result = bulk_update_embedding(db, "embeddings", ids, new_embs)
println("Bulk update result: ", result)
"""
function bulk_update_embedding(db::SQLite.DB, collection_name::String, 
    id_texts::Vector{<:AbstractString}, new_embeddings::Vector{<:AbstractVector{<:Number}};
    data_type::Union{Nothing,String}=nothing)

    n = length(id_texts)
    if n > BULK_LIMIT
        error("Bulk update limit exceeded: $n > $BULK_LIMIT")
    end
    if length(new_embeddings) != n
        error("Mismatch between number of IDs and new embeddings")
    end
    embedding_length = length(new_embeddings[1])
    for e in new_embeddings
        if length(e) != embedding_length
            error("All embeddings must have the same length")
        end
    end
    if data_type === nothing
        data_type = infer_data_type(new_embeddings[1])
    end

    SQLite.transaction(db) do
        stmt = """
        UPDATE $(collection_name)
        SET embedding_json = ?, data_type = ?
        WHERE id_text = ?
        """
        for i in 1:n
            emb_json = to_json_embedding(new_embeddings[i])
            SQLite.execute(db, stmt, (emb_json, data_type, string(id_texts[i])))
        end
    end
    return "true"
end

""" 
bulk_update_embedding(db_path::String, collection_name::String, id_texts::Vector{<:AbstractString}, new_embeddings::Vector{<:AbstractVector{<:Number}}; data_type::Union{Nothing,String}=nothing) -> Union{String}

A convenience wrapper for performing a bulk update of embedding vectors using a database file path.

This function opens the SQLite database located at db_path and then calls the primary bulk_update_embedding function to update the embeddings for the specified identifiers in the given collection (table). The function ensures that the database connection is properly closed after the operation. In case an error occurs during the update, a descriptive error message is returned as a String.

# Arguments
- db_path::String: The file path to the SQLite database.
- collection_name::String: The name of the table containing the embeddings.
- id_texts::Vector{<:AbstractString}: A vector of identifier strings corresponding to the rows to update.
- new_embeddings::Vector{<:AbstractVector{<:Number}}: A vector of new embedding vectors. All embeddings must be of the same length.
- data_type::Union{Nothing, String}=nothing: Optional. A string representing the data type of the embeddings. If omitted, the data type is inferred from the first embedding.

# Returns
- On success: A String ("true") indicating that the update was successful.
- On error: A String containing the error message.

# Example
```julia
result = bulk_update_embedding("mydatabase.sqlite", "embeddings", ["id1", "id2"], [[1.0, 2.0], [3.0, 4.0]])
if startswith(result, "Error:")
    println("Bulk update failed: ", result)
else
    println("Bulk update successful: ", result)
end
"""
function bulk_update_embedding(db_path::String, collection_name::String, 
    id_texts::Vector{<:AbstractString}, new_embeddings::Vector{<:AbstractVector{<:Number}};
    data_type::Union{Nothing,String}=nothing)
    db = open_db(db_path)
    try
        msg = bulk_update_embedding(db, collection_name, id_texts, new_embeddings; data_type=data_type)
        close_db(db)
        return msg
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
module WunDeeDB

using SQLite
using Tables, DBInterface
using JSON3


export get_supported_data_types, get_supported_endianness_types
        initialize_db, open_db, close_db, delete_db, delete_all_embeddings,
        to_json_embedding, infer_data_type,
        insert_embedding, #TODO: make plural
        delete_embedding, #TODO: make plural 
        update_embedding, #TODO: make plural 
        get_embedding, #TODO: make plural
        random_embeddings, 
        get_next_id, get_previous_id,
        count_entries, 
        get_embedding_size #TODO: get meta data fn?/?

        

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

#TODO: non-public docs
function parse_data_type(dt::String) 
    T = get(DATA_TYPE_MAP, dt, nothing) 
    if T === nothing 
        error("Unsupported data type: $dt") 
    end 
    
    return T 
end

#TODO: non-public docs
function blob_to_embedding(blob::Vector{UInt8}, ::Type{T}) where T
    # reinterpret produces a view; use collect to obtain a standard Julia array.
    return collect(reinterpret(T, blob))
end


# TODO: public doc string
function get_embedding(db::SQLite.DB, id_input)
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
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, stmt, params)))
    
    #get meta table information to determine the stored data type
    meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, META_SELECT_ALL_QUERY)))
    if isempty(meta_rows)
        error("Meta table is empty. The meta row should have been initialized during database setup.")
    end
    meta = meta_rows[1]
    dt_string = meta.data_type
    T = parse_data_type(dt_string)
    
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

# TODO: public doc string
function get_embedding(db_path::String, id_input)
    db = open_db(db_path)
    try
        result = get_embedding(db, id_input)
        close_db(db)
        return result
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end



# TODO: public doc string
function random_embeddings(db::SQLite.DB, num::Int)
    # TODO: global stmt
    stmt = """
        SELECT id_text, embedding_blob
        FROM $MAIN_TABLE_NAME
        ORDER BY RANDOM()
        LIMIT ?;
    """
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, stmt, (num,))))
    
    #meta table info to determine the stored data type
    meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, META_SELECT_ALL_QUERY)))
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

# TODO: public doc string
function random_embeddings(db_path::String, num::Int)
    db = open_db(db_path)
    try
        result = random_embeddings(db, num)
        close_db(db)
        return result
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end





#TODO: public doc string
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

    #make the SQL query
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

    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, query, (string(current_id),))))
    if isempty(rows)
        return nothing
    end
    row = rows[1]
    
    if !full_row
        return row.id_text
    else
        # For full_row retrieval, obtain the stored data type from the meta table.
        meta_rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, META_SELECT_ALL_QUERY)))
        if isempty(meta_rows)
            error("Meta table is empty. The meta row should have been initialized during database setup.")
        end
        meta = meta_rows[1]
        T = parse_data_type(meta.data_type)
        embedding_vec = blob_to_embedding(row.embedding_blob, T)
        return (id_text = row.id_text, embedding = embedding_vec, data_type = meta.data_type)
    end
end

#TODO: public doc string
function get_adjacent_id(db_path::String, current_id; direction="next", full_row=true)
    db = open_db(db_path)
    try
        result = get_adjacent_id(db, current_id; direction=direction, full_row=full_row)
        close_db(db)
        return result
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end











#TODO: public doc string
function count_entries(db::SQLite.DB; update_meta::Bool=false)
    stmt = "SELECT COUNT(*) AS count FROM $MAIN_TABLE_NAME"
    #the Tables interface for consistent row conversion
    rows = collect(Tables.namedtupleiterator(SQLite.execute(db, stmt)))
    count = rows[1].count

    if update_meta
        if count > 0
            update_stmt = "UPDATE $META_DATA_TABLE_NAME SET embedding_count = ?"
            SQLite.execute(db, update_stmt, (count,))
        else
            #if count is 0, clear the meta information (embedding_count = 0 and embedding_length = NULL)
            update_stmt = "UPDATE $META_DATA_TABLE_NAME SET embedding_count = 0"
            SQLite.execute(db, update_stmt)
        end
    end

    return count
end

#TODO: public doc string
function count_entries(db_path::String; update_meta::Bool=false)
    db = open_db(db_path)
    try
        count = count_entries(db; update_meta=update_meta)
        close_db(db)
        return count
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end




#TODO: public doc string
function get_meta_data(db::SQLite.DB)
    stmt = "SELECT * FROM $META_DATA_TABLE_NAME"
    rows = collect(Tables.namedtupleiterator(SQLite.execute(db, stmt)))
    if isempty(rows)
        return nothing
    else
        return rows[1]
    end
end

#TODO: public doc string
function get_meta_data(db_path::String)
    db = open_db(db_path)
    try
        result = get_meta_data(db)
        close_db(db)
        return result
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end









end #END MODULE
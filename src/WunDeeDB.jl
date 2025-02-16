module WunDeeDB

using SQLite, JSON3, Tables

export initialize_db, 
        open_db, close_db,
        insert_embedding, bulk_insert_embedding, 
        delete_embedding, bulk_delete_embedding, 
        update_embedding, bulk_update_embedding,
        get_embedding, bulk_get_embedding,
        get_next_id, get_previous_id

        

###################
#TODO:
#linear exact search for Retrieval
#parallelize the to and from JSON for the embeddings on bulk?     params = [(string(id_texts[i]), JSON3.write(embeddings[i]), data_type) for i in 1:n]
###################

const BULK_LIMIT = 1000

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


# DB connection fns 
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

function close_db(db::SQLite.DB)
    SQLite.close(db)
end


# Initialization 
function initialize_db(db_path::String, collection_name::String)
    db = open_db(db_path)

    try
        SQLite.execute(db, "PRAGMA journal_mode = WAL;")
        SQLite.execute(db, "PRAGMA synchronous = NORMAL;")

        create_main_stmt = """
        CREATE TABLE IF NOT EXISTS $collection_name (
            id_text TEXT PRIMARY KEY,
            embedding_json TEXT NOT NULL,
            data_type TEXT
        )
        """
        SQLite.execute(db, create_main_stmt)

        #meta table
        meta_stmt = """
        CREATE TABLE IF NOT EXISTS $(collection_name)_meta (
            row_num BIGINT,
            vector_length INT
        )
        """
        SQLite.execute(db, meta_stmt)

        close_db(db)
        return "true"
    catch e
        close_db(db)
        return "Error: $(e)"
    end
end



# Serialization Helpers
function to_json_embedding(vec::AbstractVector{<:Number})
    return JSON3.write(vec)
end

function infer_data_type(embedding::AbstractVector{<:Number}) 
    elty = eltype(embedding)
    return string(elty) 
end



# Meta Table Helpers
function update_meta(db::SQLite.DB, meta_table::String, embedding_length::Int)
    q_str = "SELECT row_num, vector_length FROM $(meta_table)"
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, q_str)))
    if isempty(rows)
        stmt = """
        INSERT INTO $(meta_table) (row_num, vector_length)
        VALUES (?, ?)
        """
        SQLite.execute(db, stmt, (1, embedding_length))
    else
        row = rows[1]
        old_row_num = row.row_num
        old_vec_len = row.vector_length
        new_row_num = old_row_num + 1
        if old_vec_len != embedding_length
            error("Vector length mismatch: existing=$(old_vec_len), new=$(embedding_length).")
        end
        stmt = "UPDATE $(meta_table) SET row_num = ?"
        SQLite.execute(db, stmt, (new_row_num,))
    end
end


# Bulk version: add `count` rows
function update_meta_bulk(db::SQLite.DB, meta_table::String, embedding_length::Int, count::Int)
    q_str = "SELECT row_num, vector_length FROM $(meta_table)"
    #use DBInterface.execute to obtain a result set that supports the Tables interface!!!!
    rows = collect(Tables.namedtupleiterator(DBInterface.execute(db, q_str)))
    if isempty(rows)
        stmt = """
        INSERT INTO $(meta_table) (row_num, vector_length)
        VALUES (?, ?)
        """
        DBInterface.execute(db, stmt, (count, embedding_length))
    else
        row = rows[1]
        old_row_num = row.row_num
        old_vec_len = row.vector_length
        if old_vec_len != embedding_length
            error("Vector length mismatch in meta: existing=$old_vec_len, new=$embedding_length")
        end
        new_row_num = old_row_num + count
        stmt = "UPDATE $(meta_table) SET row_num = ?"
        DBInterface.execute(db, stmt, (new_row_num,))
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
    rows = collect(SQLite.execute(db, q_str))
    if isempty(rows)
        return
    end
    row = rows[1]
    old_row_num = row.row_num
    new_row_num = max(old_row_num - count, 0)
    if new_row_num == 0
        stmt = "UPDATE $(meta_table) SET row_num = 0, vector_length = NULL"
        SQLite.execute(db, stmt)
    else
        stmt = "UPDATE $(meta_table) SET row_num = ?"
        SQLite.execute(db, stmt, new_row_num)
    end
end



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
    params = [(string(id_texts[i]), JSON3.write(embeddings[i]), data_type) for i in 1:n]
    SQLite.transaction(db) do
        stmt = """
        INSERT INTO $collection_name (id_text, embedding_json, data_type)
        VALUES (?, ?, ?)
        """
        SQLite.execute(db, stmt, params)
        update_meta_bulk(db, "$(collection_name)_meta", embedding_length, n)
    end
    return "true"
end

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



function delete_embedding(db::SQLite.DB, collection_name::String, id_text)
    SQLite.transaction(db) do
        
        check_sql = """
        SELECT 1 FROM $collection_name WHERE id_text = ?
        """
        check_iter = SQLite.execute(db, check_sql, (string(id_text),))
        found_rows = collect(check_iter)
        if isempty(found_rows)
            error("notfound")
        end
        
        delete_sql = """
        DELETE FROM $collection_name
        WHERE id_text = ?
        """
        SQLite.execute(db, delete_sql, (string(id_text),))
        
        update_meta_delete(db, "$(collection_name)_meta")
    end
    return "true"
end

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


function bulk_delete_embedding(db::SQLite.DB, collection_name::String, id_texts::Vector{<:AbstractString})
    n = length(id_texts)
    if n > BULK_LIMIT
        error("Bulk delete limit exceeded: $n > $BULK_LIMIT")
    end
    SQLite.transaction(db) do
        
        placeholders = join(fill("?", n), ", ")
        stmt = "DELETE FROM $collection_name WHERE id_text IN ($placeholders)"
        params = Tuple(string.(id_texts))
        SQLite.execute(db, stmt, params)
        
        bulk_update_meta_delete(db, "$(collection_name)_meta", n)
    end
    return "true"
end

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
function update_embedding(db::SQLite.DB, collection_name::String, id_text, new_embedding::AbstractVector{<:Number};
    data_type::Union{Nothing,String}=nothing)
    SQLite.transaction(db) do
        check_sql = """
        SELECT 1 FROM $collection_name WHERE id_text = ?
        """
        rows_found = collect(SQLite.execute(db, check_sql, (string(id_text),)))
        if isempty(rows_found)
            error("notfound")
        end
        meta_table = "$(collection_name)_meta"
        meta_sql = "SELECT vector_length FROM $meta_table"
        meta_rows = collect(SQLite.execute(db, meta_sql))
        if isempty(meta_rows)
            throw("No metadata found in $meta_table; can't validate dimension.")
        end
        stored_length = meta_rows[1].vector_length
        new_length = length(new_embedding)
        if stored_length != new_length
            throw("Vector length mismatch: stored=$stored_length, new=$new_length")
        end
        emb_json = to_json_embedding(new_embedding)
        if data_type === nothing
            data_type = infer_data_type(new_embedding)
        end
        update_sql = """
        UPDATE $collection_name
        SET embedding_json = ?, data_type = ?
        WHERE id_text = ?
        """
        SQLite.execute(db, update_sql, (emb_json, data_type, string(id_text)))
    end
    return "true"
end

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
        UPDATE $collection_name
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

function get_embedding(db::SQLite.DB, collection_name::String, id_text) 
    sql = """ 
    SELECT embedding_json, data_type 
    FROM $collection_name 
    WHERE id_text = ? 
    """ 
    rows = collect(SQLite.execute(db, sql, (string(id_text),))) 
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

function bulk_get_embedding(db::SQLite.DB, collection_name::String, id_texts::Vector{<:AbstractString})
    n = length(id_texts)
    if n > BULK_LIMIT
        error("Bulk get limit exceeded: $n > $BULK_LIMIT")
    end
    placeholders = join(fill("?", n), ", ")
    stmt = """
    SELECT id_text, embedding_json, data_type FROM $collection_name
    WHERE id_text IN ($placeholders)
    """
    params = Tuple(string.(id_texts))
    rows = collect(SQLite.execute(db, stmt, params))
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

function get_next_id(db::SQLite.DB, collection_name::String, current_id; full_row::Bool=false)
    if full_row
        query = """
            SELECT id_text, embedding_json, data_type
            FROM $collection_name
            WHERE id_text > ?
            ORDER BY id_text ASC
            LIMIT 1;
        """
    else
        query = """
            SELECT id_text
            FROM $collection_name
            WHERE id_text > ?
            ORDER BY id_text ASC
            LIMIT 1;
        """
    end
    rows = collect(SQLite.execute(db, query, (current_id,)))
    if isempty(rows)
        return nothing
    end
    row = rows[1]
    if !full_row
        return row.id_text
    else
        # Parse the JSON embedding using the stored data type.
        T = parse_data_type(row.data_type)
        embedding_vec = JSON3.read(row.embedding_json, Vector{T})
        return (id_text = row.id_text, embedding = embedding_vec, data_type = row.data_type)
    end
end

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

function get_previous_id(db::SQLite.DB, collection_name::String, current_id; full_row::Bool=false)
    if full_row
        query = """
            SELECT id_text, embedding_json, data_type
            FROM $collection_name
            WHERE id_text < ?
            ORDER BY id_text DESC
            LIMIT 1;
        """
    else
        query = """
            SELECT id_text
            FROM $collection_name
            WHERE id_text < ?
            ORDER BY id_text DESC
            LIMIT 1;
        """
    end
    rows = collect(SQLite.execute(db, query, (current_id,)))
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

function count_entries(db::SQLite.DB, collection_name::String; update_meta::Bool=false)
    stmt = "SELECT COUNT(*) AS count FROM $collection_name"
    rows = collect(SQLite.execute(db, stmt))
    count = rows[1].count
    
    if update_meta
        if count > 0
            update_stmt = "UPDATE $(collection_name)_meta SET row_num = ?"
            SQLite.execute(db, update_stmt, (count,))
        else
            #if count is 0, clear the meta information.
            update_stmt = "UPDATE $(collection_name)_meta SET row_num = 0, vector_length = NULL"
            SQLite.execute(db, update_stmt)
        end
    end
    
    return count
end

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


function get_embedding_size(db::SQLite.DB, collection_name::String)
    stmt = "SELECT vector_length FROM $(collection_name)_meta"
    rows = collect(SQLite.execute(db, stmt))
    if isempty(rows)
        return 0
    else
        return rows[1].vector_length
    end
end

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


function random_embeddings(db::SQLite.DB, collection_name::String, num::Int)

    if num < 1 || num > BULK_LIMIT
        error("Requested number of random embeddings must be between 1 and $BULK_LIMIT")
    end

    stmt = """
        SELECT id_text, embedding_json, data_type 
        FROM $collection_name 
        ORDER BY RANDOM() 
        LIMIT ?;
    """
    rows = collect(SQLite.execute(db, stmt, (num,)))
    
    results = Vector{Dict{String,Any}}(undef, length(rows))

    for (i, row) in enumerate(rows)
        T = parse_data_type(row.data_type)
        embedding = JSON3.read(row.embedding_json, Vector{T})
        results[i] = Dict("id_text" => row.id_text, "embedding" => embedding, "data_type" => row.data_type)
    end

    return results
end

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
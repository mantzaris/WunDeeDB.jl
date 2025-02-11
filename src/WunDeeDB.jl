module WunDeeDB

using SQLite, JSON3

export initialize_db, insert_embedding, new_add

"""
This function adds two numbers.

# Example(s)
    julia> new_add(2,3)
    5
"""
function new_add(a, b)
    return a + b
end


####################
#TODO:
#transactions and atomicity, SQLite.transaction(db) do...
#linear exact search for Retrieval
#make CRUD SQL statements constants global
###################

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



function open_db(db_path::String)
    dirpath = Base.dirname(db_path)
    
    if !isdir(dirpath)
        try
            mkpath(dirpath)
        catch e
            error("ERROR: Could not create directory $dirpath. Original error: $(e)")
        end
    end
    return SQLite.DB(db_path)
end

function close_db(db::SQLite.DB)
    SQLite.close(db)
end


function initialize_db(db_path::String, collection_name::String)
    db = open_db(db_path)

    try
        #main embeddings table
        create_main_stmt = """
        CREATE TABLE IF NOT EXISTS $collection_name (
            id_text TEXT PRIMARY KEY,
            embedding_blob BLOB NOT NULL,
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


function to_blob(vec::AbstractVector{<:Number})    
    return Vector{UInt8}(reinterpret(UInt8, vec))
end

function infer_data_type(embedding::AbstractVector{<:Number})
    elty = eltype(embedding)
    return string(elty)
end


function update_meta(db::SQLite.DB, collection_name::String, embedding_length::Int)
    q_str = "SELECT row_num, vector_length FROM $(collection_name)_meta"
    
    result_iterator = SQLite.execute(db, q_str)
    rows = collect(result_iterator)

    if isempty(rows) #if no rows we set row_num=1, vector_length=embedding_length
        insert_str = """
        INSERT INTO $(collection_name)_meta (row_num, vector_length)
        VALUES (?, ?)
        """
        SQLite.execute(db, insert_str, (1, embedding_length))
    else #there is at least one row; we'll consider only the first
        row = rows[1]
        old_row_num = row.row_num
        old_vec_len = row.vector_length

        new_row_num = old_row_num + 1

        #check dimension consistency
        if old_vec_len != embedding_length
            throw("Vector length mismatch: existing=$old_vec_len, new=$embedding_length.")
        end

        update_str = """
        UPDATE $(collection_name)_meta
        SET row_num = ?
        """
        SQLite.execute(db, update_str, new_row_num)
    end
end

function insert_embedding(db::SQLite.DB,
    collection_name::String,
    id_text,
    embedding::AbstractVector{<:Number};
    data_type::Union{Nothing,String}=nothing)

    if data_type === nothing
        data_type = infer_data_type(embedding)
    end

    emb_blob = to_blob(embedding)

    stmt = """
    INSERT INTO $collection_name (id_text, embedding_blob, data_type)
    VALUES (?, ?, ?)
    """
    SQLite.execute(db, stmt, (string(id_text), emb_blob, data_type))

    return "true"
end

function insert_embedding(db_path::String,
    collection_name::String,
    id_text,
    embedding::AbstractVector{<:Number};
    data_type::Union{Nothing,String}=nothing)

    db = open_db(db_path)
    try
        msg = insert_embedding(db, collection_name, id_text, embedding; data_type=data_type)
        update_meta(db, "$(collection_name)_meta", length(embedding) )

        close_db(db)
        return msg
    catch e
        close_db(db)
        "Error: $(e)"
    end
end



#DELETE
function update_meta_delete(db::SQLite.DB, meta_table::String)
    q_str = "SELECT row_num, vector_length FROM $meta_table"
    rows = collect(SQLite.execute(db, q_str))
    if isempty(rows)
        return
    end

    row = rows[1]
    old_row_num = row.row_num
    old_vec_len = row.vector_length

    new_row_num = max(old_row_num - 1, 0)

    if new_row_num == 0
        update_str = "UPDATE $meta_table SET row_num = 0, vector_length = NULL"
        SQLite.execute(db, update_str)
    else
        update_str = "UPDATE $meta_table SET row_num = ?"
        SQLite.execute(db, update_str, new_row_num)
    end
end

function delete_embedding(db::SQLite.DB, collection_name::String, id_text)
    check_sql = """
    SELECT 1 FROM $collection_name WHERE id_text = ?
    """
    check_iter = SQLite.execute(db, check_sql, (string(id_text),))
    found_rows = collect(check_iter)

    if isempty(found_rows)
        return "notfound"
    end

    delete_sql = """
    DELETE FROM $collection_name
    WHERE id_text = ?
    """
    SQLite.execute(db, delete_sql, (string(id_text),))

    update_meta_delete(db, "$(collection_name)_meta")

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



#UPDATE
function update_embedding(db::SQLite.DB,
    collection_name::String,
    id_text,
    new_embedding::AbstractVector{<:Number};
    data_type::Union{Nothing,String}=nothing)

    check_sql = """
    SELECT 1 FROM $collection_name WHERE id_text = ?
    """
    rows_found = collect(SQLite.execute(db, check_sql, (string(id_text),)))
    if isempty(rows_found)
        return "notfound"
    end

    meta_table = "$(collection_name)_meta"
    meta_sql = "SELECT vector_length FROM $meta_table"
    meta_rows = collect(SQLite.execute(db, meta_sql))
    if isempty(meta_rows)
        # no meta row => either error or allow any dimension
        throw("No metadata found in $meta_table; can't validate dimension.")
    end

    stored_length = meta_rows[1].vector_length
    new_length = length(new_embedding)
    if stored_length != new_length
        throw("Vector length mismatch: stored=$stored_length, new=$new_length")
    end

    emb_blob = to_blob(new_embedding)

    if data_type === nothing
        data_type = infer_data_type(new_embedding)
    end

    update_sql = """
    UPDATE $collection_name
    SET embedding_blob = ?, data_type = ?
    WHERE id_text = ?
    """
    SQLite.execute(db, update_sql, (emb_blob, data_type, string(id_text)))

    return "true"
end

function update_embedding(db_path::String,
    collection_name::String,
    id_text,
    new_embedding::AbstractVector{<:Number};
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
    SELECT embedding_blob, data_type
    FROM $collection_name
    WHERE id_text = ?
    """
    rows = collect(SQLite.execute(db, sql, (string(id_text),)))
    if isempty(rows)
        return nothing
    end

    row = rows[1]
    raw_bytes = row.embedding_blob  #an embedding stored as a 'Vector{UInt8}'
    dt_string = row.data_type       #data_type for it can be eg "Float32" or "Float64"

    T = parse_data_type(dt_string)  #get the type from the string label

    emb_reinterpreted = reinterpret(T, raw_bytes)
    embedding_vec = collect(emb_reinterpreted)

    return embedding_vec
end

function get_embedding(db_path::String, collection_name::String, id_text)
    db = open_db(db_path)
    try
        vec = get_embedding(db, collection_name, id_text)
        close_db(db)
        return vec  # either nothing or a Vector{T}
    catch e
        close_db(db)
        # Return an error string, or rethrow
        return "Error: $(e)"
    end
end


end #END MODULE
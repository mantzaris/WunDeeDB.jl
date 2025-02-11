module WunDeeDB

using SQLite

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


#Delete
#TODO remove collection

#update
#TODO update entry

#Retrieve, up to how many as well, remember to reconvert data back to original
function get_embedding(db_path::String, collection_name::String, id)
    db = SQLite.DB(db_path)
    try
        q = SQLite.execute(db, "SELECT embedding FROM $collection_name WHERE id = ?", string(id))
        rows = collect(q)
        SQLite.close(db)
        if length(rows) == 0
            return nothing
        else
           
            raw_bytes = rows[1].embedding
            
            emb_f32 = reinterpret(Float32, raw_bytes)
            return copy(emb_f32)
        end
    catch e
        SQLite.close(db)
        rethrow(e)
    end
end


end
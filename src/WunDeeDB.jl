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


# TODO: add transactions

#Initialize (or open) the database file at `db_path`. If the file's directory does not exist, return a warning message.
#Ensures a table named `collection_name` exists with:'id TEXT PRIMARY KEY','embedding BLOB NOT NULL'
#Also creates a metadata table named `collection_name_meta` to store the vector length, unless it already exists.
#Returns: A `String` indicating success or a warning message if the directory doesn't exist
function initialize_db(db_path::String, collection_name::String, vector_length::Int)
    dirpath = Base.dirname(db_path)

    if !isdir(dirpath)
        try
            mkpath(dirpath)
        catch e
            return "ERROR: Could not create directory $dirpath. Original error: $(e)"
        end
    end

    db = SQLite.DB(db_path)
    try
        create_stmt = """
        CREATE TABLE IF NOT EXISTS $collection_name (
            id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL
        )
        """
        SQLite.execute(db, create_stmt)

        # TODO: now make it that it stores the embedding type
        meta_stmt = """
        CREATE TABLE IF NOT EXISTS ${collection_name}_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
        SQLite.execute(db, meta_stmt)

        insert_meta_stmt = """
        INSERT OR IGNORE INTO ${collection_name}_meta (key, value)
        VALUES ('vector_length', ?)
        """
        SQLite.execute(db, insert_meta_stmt, string(vector_length))

        # 6. Close DB and return success
        SQLite.close(db)
        return "Database initialized successfully at '$db_path' with table '$collection_name'."
    catch e
        SQLite.close(db)
        rethrow(e)
    end
end


#Create
function insert_embedding(db_path::String,
                            collection_name::String,
                            id,
                            embedding::AbstractVector{<:Number})
    db = SQLite.DB(db_path)
    try

        emb_blob = to_blob(embedding)

        stmt = "INSERT INTO $collection_name (id, embedding) VALUES (?, ?)"
        SQLite.execute(db, stmt, (string(id), emb_blob))

        SQLite.close(db)
        return "Embedding inserted successfully for id = $(string(id))."
        catch e
        SQLite.close(db)
        rethrow(e)
    end
end

function to_blob(vec::AbstractVector{<:Number})
    
    return Vector{UInt8}(reinterpret(UInt8, vec))
end

#Delete
#TODO remove collection

#update
#TODO update entry

#Retrieve, up to how many as well, remember to reconvert data back to original
function get_embedding(db_path::String, collection_name::String, id)
    db = SQLite.DB(db_path)
    try
        q = SQLite.Query(db, "SELECT embedding FROM $collection_name WHERE id = ?", string(id))
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
#https://github.com/sadit/SimilaritySearch.jl
#https://sadit.github.io/SimilaritySearch.jl/dev/api/

using HDF5, SimilaritySearch

# keep_conn_open ? KEEP_DB_OPEN[] = true : KEEP_DB_OPEN[] = false
# db = !isnothing(DB_HANDLE[]) ? DB_HANDLE[] : open_db(db_path)

# const DB_HANDLE = Ref{Union{SQLite.DB, Nothing}}(nothing) #handle for the db connection to use instead of open/close fast
# const KEEP_DB_OPEN = Ref{Bool}(true)

# const H5_PATH = "my_vector_db.h5"

# # Ensure the HDF5 file is initialized.
# VectorDB.initialize_vector_db(H5_PATH)

# # Insert a new vector.
# embedding = rand(Float32, 128)
# new_pos = VectorDB.insert_vector!(H5_PATH, embedding)

# # Search for similar vectors.
# results = VectorDB.search_vectors(H5_PATH, embedding, 5)

# # Delete a vector (by its position).
# VectorDB.delete_vector!(H5_PATH, new_pos)


function initialize_vector_db(h5_path::String)
    if !isfile(h5_path)
        h5open(h5_path, "w") do h5file
            # Create an empty dataset; dimensions will be set on first insertion.
            create_dataset(h5file, "vectors", Float32, ((0, 0), (-1, -1)), chunk=(1024, 1024))
        end
    end
end

function _load_or_create_index(h5_path::String, vector_dim::Int)
    db = nothing
    index = nothing
    h5open(h5_path, "r") do h5file
        ds = h5file["vectors"]
        # Create a MatrixDatabase using the dataset.
        db = MatrixDatabase(ds)
        if haskey(h5file, "index")
            # Read the binary data for the serialized index.
            buf = IOBuffer(read(h5file["index"]))
            index = deserialize(buf)
            # Reattach the current database.
            index.db = db
        else
            # Create a new index and build it.
            index = SearchGraph(; dist=SqL2Distance(), db=db)
            index!(index)
        end
    end
    return index
end

function _save_index(h5_path::String, index::SearchGraph)
    h5open(h5_path, "r+") do h5file
        buf = IOBuffer()
        serialize(buf, index)
        # Remove any existing "index" object to avoid conflicts.
        if haskey(h5file, "index")
            delete_object(h5file, "index")
        end
        # Write the binary data back into the HDF5 file.
        h5file["index"] = take!(buf)
    end
end

function insert_vector!(h5_path::String, embedding::Vector{Float32})
    cur_size = 0
    h5open(h5_path, "r+") do h5file
        ds = h5file["vectors"]
        cur_size = size(ds, 2)
        # On first insertion, set the dataset dimension to match the embedding length.
        if size(ds, 1) == 0
            set_extent_dims(ds, (length(embedding), 0))
        end
        set_extent_dims(ds, (length(embedding), cur_size + 1))
        ds[:, cur_size + 1] = embedding
    end

    # Load (or create) the index, update it, and save it back.
    index = _load_or_create_index(h5_path, length(embedding))
    push!(index, embedding)
    _save_index(h5_path, index)
    
    return cur_size + 1
end

function delete_vector!(h5_path::String, pos::Int)
    vector_dim = 0
    h5open(h5_path, "r") do h5file
        ds = h5file["vectors"]
        vector_dim = size(ds, 1)
    end
    index = _load_or_create_index(h5_path, vector_dim)
    delete!(index, pos)
    _save_index(h5_path, index)
end

function search_vectors(h5_path::String, query::Vector{Float32}, k::Int)
    index = _load_or_create_index(h5_path, length(query))
    res = search(index, query, k)
    return [(id, dist) for (id, dist) in res]
end

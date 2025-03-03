
"""
    get_supported_data_types() -> Vector{String}

Returns a sorted vector of supported data type names as strings.
"""
function get_supported_data_types()::Vector{String}
    return sort(collect(keys(DATA_TYPE_MAP)))
end



function parse_data_type(dt::String) 
    T = get(DATA_TYPE_MAP, dt, nothing) 
    if T === nothing 
        error("Unsupported data type: $dt") 
    end 
    
    return T 
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
```
"""
function infer_data_type(embedding::AbstractVector{<:Number})
    return string(eltype(embedding))
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


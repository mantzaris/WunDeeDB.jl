```@meta
CurrentModule = WunDeeDB
```

```@contents

```


# WunDeeDB

WunDeeDB is a Julia package for **storing and querying embeddings** in a SQLite database. It supports a variety of numerical types (including `Float16`, `Float32`, `Float64`, and various integer types) and can integrate with approximate nearest neighbor indices (like **HNSW** or **LM‐DiskANN**). By default, the module provides a **linear fallback** for distance queries if no ANN method is enabled.

This module exposes **bulk operations** for insertions, deletions, and updates (capped at a certain transaction size) and provides convenient high‐level functions (`insert_embeddings`, `delete_embeddings`, `search_ann`, etc.) so users can focus on storing and retrieving embeddings without worrying about the underlying details of adjacency structures. Whether you want a simple linear search or a more scalable approach with HNSW or LM‐DiskANN, WunDeeDB automatically handles the index creation and updates.

Internally, WunDeeDB stores embedding vectors in a `BLOB` column, along with metadata (`embedding_length`, `data_type`) in a meta table to ensure consistency. When you choose `ann="hnsw"` or `ann="lmdiskann"`, each call to `insert_embeddings` automatically triggers adjacency insertion for that node, enabling fast approximate searches for new queries via `search_ann`.

---

## Supported Data Types

WunDeeDB supports storing embeddings of these types (any dimension in principle):

Float16, Float32, Float64, BigFloat, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Int128, UInt128




---

## Exported Functions

**General DB**  
- `initialize_db`, `open_db`, `close_db`, `delete_db`, `delete_all_embeddings`  
- `get_meta_data`, `update_description`, `count_entries`

**Embeddings**  
- `insert_embeddings`, `delete_embeddings`, `update_embeddings`  
- `get_embeddings`, `get_all_ids`, `get_all_embeddings`, `random_embeddings`

**ANN / Searching**  
- `search_ann` (resolves to HNSW, LM‐DiskANN, or linear fallback)  
- `supported_distance_metrics`

**Linear Search**  
- `linear_search_all_embeddings`, `linear_search_ids`,  
  `linear_search_iteration`, `linear_search_ids_batched`

---

## Quick Example (HNSW, LM‐DiskANN, & Linear)

Below is a **minimal code snippet** showing how to create a database, insert a couple embeddings, and then run searches with **HNSW**, **LM‐DiskANN**, and a **linear** fallback. (We assume each approach uses a separate DB for illustration.)

Install via: `(@v1.9) pkg> add https://github.com/mantzaris/WunDeeDB.jl`

```julia
using WunDeeDB

#
# 1) Example: HNSW
#
hnsw_db = "temp_hnsw.sqlite"
initialize_db(hnsw_db, 3, "Float32"; ann="hnsw")
insert_embeddings(hnsw_db, "node1", Float32[0.0, 0.0, 0.0])
insert_embeddings(hnsw_db, "node2", Float32[1.0, 1.0, 1.0])

# search for top-1 neighbor using HNSW adjacency
found_hnsw = search_ann(hnsw_db, Float32[0.1, 0.1, 0.1], "euclidean"; top_k=1)
println("HNSW found: ", found_hnsw)

#
# 2) Example: LM-DiskANN
#
lmdiskann_db = "temp_lmdiskann.sqlite"
initialize_db(lmdiskann_db, 3, "Float32"; ann="lmdiskann")
insert_embeddings(lmdiskann_db, "nodeA", Float32[0.5, 0.5, 0.4])
insert_embeddings(lmdiskann_db, "nodeB", Float32[0.8, 0.9, 0.7])

# search for top-2 neighbors using LM-DiskANN adjacency
found_lmdiskann = search_ann(lmdiskann_db, Float32[0.55, 0.55, 0.35], "euclidean"; top_k=2)
println("LM-DiskANN found: ", found_lmdiskann)

#
# 3) Example: Linear fallback (no ann)
#
linear_db = "temp_linear.sqlite"
initialize_db(linear_db, 3, "Float32"; ann="")
insert_embeddings(linear_db, "X", Float32[0.0, 1.0, 2.0])
insert_embeddings(linear_db, "Y", Float32[1.0, 1.0, 2.0])

# fallback linear search:
found_linear = search_ann(linear_db, Float32[0.1, 1.0, 2.1], "euclidean"; top_k=2)
println("Linear fallback found: ", found_linear)
```

and some more examples

```julia
using WunDeeDB

#
# 1) Initialize a database, dimension=3, Float32, no ann
#
db_path = "my_demo.sqlite"
initialize_db(db_path, 3, "Float32"; keep_conn_open=true, description="Demo DB", ann="")  

#
# 2) Insert Embeddings
#    - single key => single embedding
#    - multiple keys => vector-of-vectors
#
insert_embeddings(db_path, "key1", Float32[0.1, 0.2, 0.3])
insert_embeddings(db_path, ["key2", "key3"], [Float32[1.0, 1.1, 1.2], Float32[9.0, 9.1, 9.2]])

#
# 3) Retrieve Embeddings
#    - single key => single vector
#    - multiple keys => dictionary of key => vector
#
# Single Key
my_embedding = get_embeddings(db_path, "key1")  
println("key1 embedding: ", my_embedding)

# Multiple keys
some_embeddings = get_embeddings(db_path, ["key2", "key3"])
println("Retrieved keys: ", keys(some_embeddings))
println("key2 embedding => ", some_embeddings["key2"])
println("key3 embedding => ", some_embeddings["key3"])

#
# 4) Delete Embeddings by Keys
#    - Single or multiple keys can be removed
#
delete_embeddings(db_path, "key1")
delete_embeddings(db_path, ["key2", "key3"])

# After deletion, retrieving them returns nothing or an error
check_after_delete = get_embeddings(db_path, "key1")
println("key1 after deletion => ", check_after_delete)  # likely nothing or error string

#
# 5) Close DB
#
close_db()  # or close_db(db_path) if your code does that

```

### How it works:

    Initialization: We pick ann="hnsw", ann="lmdiskann", or "" to decide which adjacency method is used.
    Insertion: Each call to insert_embeddings(db, node_id, vector) checks the ann type and updates adjacency automatically if needed.
    Search: The high‐level function search_ann(db, query, metric; top_k) uses the specified adjacency (HNSW, LM‐DiskANN, or none -> linear).

### Bulk Limit & Data Checking

WunDeeDB enforces a bulk limit of 1000 embeddings per transaction to avoid overly large inserts. You can batch multiple calls if you have more than 1000. The module also checks that the embedding dimension and data type match what’s stored in the EmbeddingsMetaData table, ensuring consistency (so you can’t insert a Float32 with length=100 into a DB initialized for Float16 length=128, etc.).

For more advanced usage (e.g., partial BFS expansions, recall tests, multi-level HNSW), see our test suite or higher-level documentation. WunDeeDB aims to handle typical ANN workflows seamlessly via a single API.


### naming of the package
In his book How JavaScript Works, Douglas Crockford advocates for spelling the word "one" as "wun" to better align with its pronunciation. He argues that the traditional spelling does not conform to standard English pronunciation rules and that having the word for 1 start with a letter resembling 0 is problematic. Since a vector database is a database for 1-D objects, it is called **Wun-Dee-DB**. 

Along with a simple name should be the simple approach for a: zero-config, embedded, WAL, just works vector database.


```@index
```

```@autodocs
Modules = [WunDeeDB]
Private = false
```
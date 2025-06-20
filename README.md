# WunDeeDB.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) 
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://mantzaris.github.io/WunDeeDB.jl/) 
[![Build Status](https://github.com/mantzaris/WunDeeDB.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mantzaris/WunDeeDB.jl/actions)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.08033/status.svg)](https://doi.org/10.21105/joss.08033)

# WunDeeDB is a vector DataBase with a SQLite backend 

([link to docs](https://mantzaris.github.io/WunDeeDB.jl))

WunDeeDB is a Julia package for **storing and querying embeddings** in a SQLite database. It supports a variety of numerical types (including `Float16`, `Float32`, `Float64`, and various integer types) and can integrate with approximate nearest neighbor indices (like **HNSW** or **LM‐DiskANN**). By default, the module provides a **linear fallback** for distance queries if no ANN method is enabled.

**HNSW**: Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE transactions on pattern analysis and machine intelligence, 42(4), 824-836.

**LM-DiskANN**: Pan, Y., Sun, J., & Yu, H. (2023, December). Lm-diskann: Low memory footprint in disk-native dynamic graph-based ann indexing. In 2023 IEEE International Conference on Big Data (BigData) (pp. 5987-5996). IEEE.

# Using the package (Quick examples for HNSW/LM-DiskANN, and linear)


**Minimal code snippet** showing how to create a database, insert a couple embeddings, and then run searches with **HNSW**, **LM‐DiskANN**, and a **linear** fallback. (We assume each approach uses a separate DB for illustration.)

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

read the documentation for more details on the variety of functions.


### naming of the package
In his book How JavaScript Works, Douglas Crockford advocates for spelling the word "one" as "wun" to better align with its pronunciation. He argues that the traditional spelling does not conform to standard English pronunciation rules and that having the word for 1 start with a letter resembling 0 is problematic. Since a vector database is a database for 1-D objects, it is called **Wun-Dee-DB**. 

Along with a simple name should be the simple approach for a: zero-config, embedded, WAL, just works vector database.

# Citing this work

- Mantzaris, A. V., (2025). WunDeeDB.jl: An easy to use, zero config, WAL, SQLite backend vector database. Journal of Open Source Software, 10(110), 8033, https://doi.org/10.21105/joss.08033

- @article{Mantzaris2025, doi = {10.21105/joss.08033}, url = {https://doi.org/10.21105/joss.08033}, year = {2025}, publisher = {The Open Journal}, volume = {10}, number = {110}, pages = {8033}, author = {Alexander V. Mantzaris}, title = {WunDeeDB.jl: An easy to use, zero config, WAL, SQLite backend vector database}, journal = {Journal of Open Source Software} } 


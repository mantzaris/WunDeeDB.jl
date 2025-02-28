var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = WunDeeDB","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#WunDeeDB","page":"Home","title":"WunDeeDB","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for WunDeeDB.","category":"page"},{"location":"#Introduction","page":"Home","title":"Introduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This module supports bulk operations (insertions, deletions, and updates) with a hard limit of 1000 records per operation. In addition, the module supports the following numerical types for embedding data:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Float16\nFloat32\nFloat64\nBigFloat\nInt8\nUInt8\nInt16\nUInt16\nInt32\nUInt32\nInt64\nUInt64\nInt128\nUInt128","category":"page"},{"location":"","page":"Home","title":"Home","text":"These types are defined in the DATATYPEMAP and are used to correctly parse and manage the embedding vectors stored in the SQLite database. The constant BULK_LIMIT is set to 1000 to prevent overly large transactions during bulk operations.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [WunDeeDB]\nPrivate = false\nOrder = [:function]","category":"page"},{"location":"#WunDeeDB.close_db-Tuple{SQLite.DB}","page":"Home","title":"WunDeeDB.close_db","text":"close_db(db::SQLite.DB)\n\nClose an open SQLite database connection.\n\nThis function is a simple wrapper around SQLite.close to ensure that the provided database connection is properly closed when it is no longer needed.\n\nArguments\n\ndb::SQLite.DB: (optional) The SQLite database connection to be closed, and if not included the default of the persistent db object is used\n\nExample\n\n```julia db = open_db(\"data/mydatabase.sqlite\")\n\nPerform database operations...\n\nclose_db(db)\n\nor  close_db()\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.count_entries-Tuple{SQLite.DB}","page":"Home","title":"WunDeeDB.count_entries","text":"Count the number of entries in the main table of the SQLite database.\n\nThis function is overloaded to support both an active database connection and a database file path:\n\ncount_entries(db::SQLite.DB; update_meta::Bool=false): Counts entries using an active database connection.\ncount_entries(db_path::String; update_meta::Bool=false): Opens the database at the specified path, counts entries, optionally updates the meta table, and then closes the connection.\n\nArguments\n\nFor count_entries(db::SQLite.DB; update_meta::Bool=false):\ndb::SQLite.DB: An active SQLite database connection.\nFor count_entries(db_path::String; update_meta::Bool=false):\ndb_path::String: The file path to the SQLite database.\nupdate_meta::Bool=false: When set to true, updates the meta table with the current count. If the count is 0, it clears the meta information (i.e., sets embedding_count to 0 and resets embedding_length if applicable).\n\nReturns\n\nThe number of entries (an integer) in the main table.\nIn the db_path overload, returns a String error message if an error occurs.\n\nExample\n\n```julia\n\nUsing an active database connection:\n\nentrycount = countentries(db, updatemeta=true) println(\"Number of entries: \", entrycount)\n\nUsing a database file path:\n\nentrycount = countentries(\"mydatabase.db\", updatemeta=true) println(\"Number of entries: \", entry_count)\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.delete_all_embeddings-Tuple{String}","page":"Home","title":"WunDeeDB.delete_all_embeddings","text":"Delete all embeddings from the database at the specified path and reset the embedding count.\n\nArguments\n\ndb_path::String: The file path of the SQLite database.\n\nReturns\n\ntrue if the operation is successful.\nA String error message if an error occurs.\n\nExample\n\n```julia result = deleteallembeddings(\"my_database.db\") if result === true     println(\"Embeddings deleted successfully.\") else     println(\"Error: 'result'\") end\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.delete_db-Tuple{String}","page":"Home","title":"WunDeeDB.delete_db","text":"Delete the database file at the specified path.\n\nArguments\n\ndb_path::String: The file path of the database to delete.\n\nReturns\n\ntrue if the file was successfully deleted.\nA String error message if deletion fails or if the file does not exist.\n\nExample\n\n```julia result = deletedb(\"mydatabase.db\") if result === true     println(\"Database deleted successfully.\") else     println(\"Error: 'result'\") end\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.delete_embeddings-Tuple{SQLite.DB, Any}","page":"Home","title":"WunDeeDB.delete_embeddings","text":"Delete one or more embeddings from the database using their ID(s).\n\nThis function is overloaded to support both an active database connection and a database file path:\n\ndelete_embeddings(db::SQLite.DB, id_input): Deletes embeddings using an open database connection.\ndelete_embeddings(db_path::String, id_input): Opens the database at the specified path, deletes the embeddings, and then closes the connection.\n\nArguments\n\nFor delete_embeddings(db::SQLite.DB, id_input):\ndb::SQLite.DB: An active SQLite database connection.\nid_input: A single ID or a collection of IDs (can be any type convertible to a string) identifying the embeddings to be deleted.\nFor delete_embeddings(db_path::String, id_input):\ndb_path::String: The file path to the SQLite database.\nid_input: A single ID or a collection of IDs identifying the embeddings to be deleted.\n\nReturns\n\ntrue if the deletion is successful.\nA String error message if an error occurs during deletion.\n\nExamples\n\nUsing an active database connection: ```julia result = delete_embeddings(db, [1, 2, 3]) if result === true     println(\"Embeddings deleted successfully.\") else     println(\"Error: \", result) end\n\nUsing a database file path:\n\nresult = deleteembeddings(\"mydatabase.db\", 1) if result === true     println(\"Embedding deleted successfully.\") else     println(\"Error: \", result) end\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.get_adjacent_id-Tuple{SQLite.DB, Any}","page":"Home","title":"WunDeeDB.get_adjacent_id","text":"Retrieve the adjacent record relative to a given current_id from the SQLite database.\n\nThis function is overloaded to support both an active database connection and a database file path:\n\nget_adjacent_id(db::SQLite.DB, current_id; direction=\"next\", full_row=true): Uses an active connection.\nget_adjacent_id(db_path::String, current_id; direction=\"next\", full_row=true): Opens the database at the specified path, retrieves the adjacent record, and then closes the connection.\n\nThe function returns the record immediately after (or before) the specified current_id based on the direction parameter. When full_row is true, the returned result is a named tuple containing the id_text, the decoded embedding vector, and the stored data_type. When full_row is false, only the id_text is returned.\n\nArguments\n\nFor get_adjacent_id(db::SQLite.DB, current_id; direction, full_row):\ndb::SQLite.DB: An active SQLite database connection.\ncurrent_id: The current record's ID from which to find the adjacent record.\ndirection::String=\"next\": The direction to search for the adjacent record. Use \"next\" for the record with an ID greater than current_id, or \"previous\" (or \"prev\") for the record with an ID less than current_id.\nfull_row::Bool=true: If true, return the full record (including embedding and meta data); if false, return only the id_text.\nFor get_adjacent_id(db_path::String, current_id; direction, full_row):\ndb_path::String: The file path to the SQLite database.\nOther parameters are as described above.\n\nReturns\n\nWhen full_row is true: A named tuple (id_text, embedding, data_type) representing the adjacent record.\nWhen full_row is false: The adjacent record's id_text.\nReturns nothing if no adjacent record is found.\nFor the db_path overload, a String error message is returned if an error occurs.\n\nExample\n\n```julia\n\nUsing an active database connection:\n\nadjacent = getadjacentid(db, 100; direction=\"previous\", full_row=false) if adjacent !== nothing     println(\"Adjacent ID: \", adjacent) else     println(\"No adjacent record found.\") end\n\nUsing a database file path:\n\nresult = getadjacentid(\"my_database.db\", 100; direction=\"next\") if result isa NamedTuple     println(\"Adjacent record: \", result) else     println(\"Error or record not found: \", result) end\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.get_all_embeddings-Tuple{SQLite.DB}","page":"Home","title":"WunDeeDB.get_all_embeddings","text":"get_all_embeddings(db::SQLite.DB)\nget_all_embeddings(db_path::String)\n\nFetches all embeddings from the main table, converting raw BLOB data into typed vectors.   Returns a Dict{String, Any} mapping each id_text to its corresponding embedding.\n\nMethod 1: get_all_embeddings(db::SQLite.DB)   Uses the provided open database connection.\n\nMethod 2: get_all_embeddings(db_path::String)   Opens a database (or reuses a global handle) from the given path. On success, returns the same dictionary of embeddings; on error, returns a string describing the error.\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.get_all_ids-Tuple{SQLite.DB}","page":"Home","title":"WunDeeDB.get_all_ids","text":"get_all_ids(db::SQLite.DB)\nget_all_ids(db_path::String)\n\nRetrieves all id_text values from the main table, returning them as a Vector{String}.\n\ngetallids(db::SQLite.DB): Uses the provided open database connection.\ngetallids(db_path::String): Opens or reuses a global DB connection from the given path; returns the same vector of IDs on success or an error string on failure.\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.get_embeddings-Tuple{SQLite.DB, Any}","page":"Home","title":"WunDeeDB.get_embeddings","text":"Retrieve one or more embeddings from the SQLite database by their ID(s).\n\nThis function is overloaded to support:\n\nget_embeddings(db::SQLite.DB, id_input): Retrieves embeddings using an active database connection.\nget_embeddings(db_path::String, id_input): Opens the database at the specified path, retrieves embeddings, and then closes the connection.\n\nWhen a single ID is provided, the corresponding embedding vector is returned (or nothing if not found). When multiple IDs are provided, a dictionary mapping each ID (as a string) to its embedding vector is returned.\n\nArguments\n\nFor get_embeddings(db::SQLite.DB, id_input):\ndb::SQLite.DB: An active SQLite database connection.\nid_input: A single ID or an array of IDs identifying the embeddings to retrieve.\nFor get_embeddings(db_path::String, id_input):\ndb_path::String: The file path to the SQLite database.\nid_input: A single ID or an array of IDs identifying the embeddings to retrieve.\n\nReturns\n\nA single embedding vector if one ID is provided, or a dictionary mapping IDs (as strings) to embedding vectors if multiple IDs are provided.\nReturns nothing if a single requested ID is not found.\nA String error message if an error occurs during retrieval.\n\nExamples\n\nUsing an active database connection: ```julia embedding = get_embeddings(db, 42) if embedding === nothing     println(\"Embedding not found.\") else     println(\"Embedding: \", embedding) end\n\nUsing a database file path:\n\nembeddings = getembeddings(\"mydatabase.db\", [1, 2, 3]) for (id, emb) in embeddings     println(\"ID: 'id', Embedding: \", emb) end\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.get_meta_data-Tuple{SQLite.DB}","page":"Home","title":"WunDeeDB.get_meta_data","text":"Retrieve meta data from the SQLite database.\n\nThis function is overloaded to accept either an active database connection or a file path:\n\nget_meta_data(db::SQLite.DB): Retrieves the meta data from the given open database connection.\nget_meta_data(db_path::String): Opens the database at the specified path, retrieves the meta data, and then closes the connection.\nneither approach will close the DB if a persistent handle is in use and tries to use the persistent one if possible.\n\nArguments\n\nFor get_meta_data(db::SQLite.DB):\ndb::SQLite.DB: An active SQLite database connection.\nFor get_meta_data(db_path::String):\ndb_path::String: The file path to the SQLite database.\n\nReturns\n\nThe first row of meta data as a named tuple if it exists, or nothing if no meta data is found.\nIf an error occurs (in the db_path overload), a String error message is returned.\n\nExamples\n\nUsing an existing database connection: ```julia meta = getmetadata(db) if meta !== nothing     println(\"Meta data: \", meta) else     println(\"No meta data available.\") end\n\nUsing a database file path:\n\nresult = getmetadata(\"my_database.db\") if result isa NamedTuple     println(\"Meta data: \", result) else     println(\"Error: \", result) end\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.get_supported_data_types-Tuple{}","page":"Home","title":"WunDeeDB.get_supported_data_types","text":"get_supported_data_types() -> Vector{String}\n\nReturns a sorted vector of supported data type names as strings.\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.infer_data_type-Tuple{AbstractVector{<:Number}}","page":"Home","title":"WunDeeDB.infer_data_type","text":"Infer the data type of the elements in a numeric embedding vector.\n\nArguments\n\nembedding::AbstractVector{<:Number}: A vector containing numerical values.\n\nReturns\n\nA String representing the element type of the embedding.\n\nExample\n\n```julia vec = [1.0, 2.0, 3.0] println(inferdatatype(vec))  # \"Float64\"\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.initialize_db-Tuple{String, Int64, String}","page":"Home","title":"WunDeeDB.initialize_db","text":"Initialize a SQLite database by setting up the main and meta tables with appropriate configuration.\n\nArguments\n\ndb_path::String: Path to the SQLite database file.\nembedding_length::Int: Length of the embedding vector. Must be 1 or greater.\ndata_type::String: Data type for the embeddings. Must be one of the supported types (use get_supported_data_types() to see valid options).\ndescription::String=\"\": (optional) User selected description meta data, defaults to empty string\nkeep_conn_open::Bool=true: (optional) Keep the DB connection open for rapid successive uses or false for multiple applications to release\n\nReturns\n\ntrue on successful initialization.\nA String error message if any parameter is invalid or if an exception occurs during initialization.\n\nExample\n\n```julia result = initializedb(\"mydatabase.db\", 128, \"float32\", description=\"embeddings from 01/01/25\", keepconnopen=true) if result === true     println(\"Database initialized successfully!\") else     println(\"Initialization failed: 'result'\") end\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.insert_embeddings-Tuple{SQLite.DB, Any, Any}","page":"Home","title":"WunDeeDB.insert_embeddings","text":"Insert one or more embeddings into a specified collection in the SQLite database.\n\nThis function is overloaded to support:\n\nActive Connection: insert_embeddings(db::SQLite.DB, id_input, embedding_input)\nDatabase Path: insert_embeddings(db_path::String, id_input, embedding_input)\n\nIn both cases, the function validates that the provided embeddings have a consistent length and that their data type matches the meta information stored in the database. For the method accepting a database path, the connection is automatically opened and closed.\n\nArguments\n\ndb::SQLite.DB or db_path::String: Either an active SQLite database connection or the file path to the database.\nid_input: A single ID or an array of IDs corresponding to the embeddings.\nembedding_input: A single numeric embedding vector or an array of embedding vectors. All embeddings must be of the same length.\n\nReturns\n\ntrue if the embeddings are successfully inserted.\nA String error message if an error occurs.\n\nExamples\n\nUsing an active database connection: ```julia result = insert_embeddings(db, [1, 2], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]) if result === true     println(\"Embeddings inserted successfully.\") else     println(\"Error: \", result) end\n\nUsing a database file path:\n\nresult = insertembeddings(\"mydatabase.db\", 1, [0.1, 0.2, 0.3]) if result === true     println(\"Embedding inserted successfully.\") else     println(\"Error: \", result) end\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.linear_search_all_embeddings-Tuple{String, AbstractVector, String}","page":"Home","title":"WunDeeDB.linear_search_all_embeddings","text":"linear_search_all_embeddings(db_path::String, query_embedding::AbstractVector, metric::String; top_k::Int=5)\n\nPerforms a brute-force linear search over all embeddings stored in the database located at db_path.   For each embedding, the function computes the distance to the provided query_embedding using the specified   metric (e.g., \"euclidean\" or \"cosine\"). It then maintains the top top_k nearest results using a max-heap   and returns a vector of tuples (distance, id_text), sorted in ascending order by distance.\n\nIf get_all_embeddings(db_path) returns an error message (as a String), this function immediately returns that error.\n\nArguments\n\ndb_path::String: Path to the SQLite database file.\nquery_embedding::AbstractVector: The query embedding vector.\nmetric::String: The distance metric to use (e.g., \"euclidean\", \"cosine\").\ntop_k::Int=5: (Optional) The number of nearest neighbors to return.\n\nReturns\n\nA vector of (distance, id_text) tuples sorted by ascending distance, or an error message String if retrieval fails.\n\nExample\n\nquery = Float32[0.5, 0.5, 0.5]\nresults = linear_search_all_embeddings(\"my_database.sqlite\", query, \"euclidean\"; top_k=3)\nfor (dist, id) in results\n    println(\"ID: \", id, \"  Distance: \", dist)\nend\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.linear_search_ids-Tuple{String, AbstractVector, String}","page":"Home","title":"WunDeeDB.linear_search_ids","text":"linear_search_ids(db_path::String, query_embedding::AbstractVector, metric::String; top_k::Int=5)\n\nPerforms a brute-force linear search by fetching all IDs at once (get_all_ids) and  computing the distance to query_embedding for each. Maintains the top_k closest  results, returning them sorted by ascending distance as tuples (distance, id_text).\n\nExample\n\nresults = linear_search_ids(\"my_database.sqlite\", my_query_embedding, \"euclidean\"; top_k=5)\nfor (dist, id) in results\n    println(\"ID: \", id, \"  Distance: \", dist)\nend\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.linear_search_ids_batched-Tuple{String, AbstractVector, String}","page":"Home","title":"WunDeeDB.linear_search_ids_batched","text":"linear_search_ids_batched(db_path::String, query_embedding::AbstractVector, metric::String; top_k::Int=5, batch_size::Int=1000)\n\nPerforms a batched brute-force linear search over embeddings stored in the database at db_path.   The function retrieves all IDs using get_all_ids and then processes the embeddings in batches   (of size batch_size). For each batch, it computes the distance from each embedding to the   query_embedding using the specified metric (e.g., \"euclidean\" or \"cosine\"). It maintains   the top top_k nearest results using a max-heap and returns a vector of tuples (distance, id_text)   sorted in ascending order by distance.\n\nIf an error occurs during the retrieval of embeddings (for example, if get_embeddings returns an   error message), the function immediately returns that error message as a String.\n\nArguments\n\ndb_path::String: Path to the SQLite database file.\nquery_embedding::AbstractVector: The query embedding vector.\nmetric::String: The distance metric to use (e.g., \"euclidean\", \"cosine\").\ntop_k::Int=5: (Optional) The number of nearest neighbors to return.\nbatch_size::Int=1000: (Optional) The number of IDs to process in each batch.\n\nReturns\n\nA vector of (distance, id_text) tuples sorted by ascending distance, or a String containing   an error message if retrieval fails.\n\nExample\n\nquery = Float32[0.5, 0.5, 0.5]\nresults = linear_search_ids_batched(\"my_database.sqlite\", query, \"euclidean\"; top_k=3, batch_size=100)\nfor (dist, id) in results\n    println(\"ID: \", id, \"  Distance: \", dist)\nend\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.linear_search_iteration-Tuple{String, AbstractVector, String}","page":"Home","title":"WunDeeDB.linear_search_iteration","text":"linear_search_iteration(db_path::String, query_embedding::AbstractVector, metric::String; top_k::Int=5)\n\nPerforms a brute-force linear search by iterating over each embedding in the database using get_adjacent_id. Computes the distance to query_embedding according to metric and returns the top top_k nearest results, sorted by ascending distance. Each result is a  tuple (distance, id_text).\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.open_db-Tuple{String}","page":"Home","title":"WunDeeDB.open_db","text":"opendb(dbpath::String) -> SQLite.DB\n\nOpen an SQLite database located at the specified file path, ensuring that the directory exists.\n\nThis function performs the following steps:\n\nDirectory Check and Creation:   It determines the directory path for db_path and checks whether it exists. If the directory does not exist, the function attempts to create it using mkpath.   If directory creation fails, an error is raised with a descriptive message.\nDatabase Connection and Configuration:   The function opens an SQLite database connection using SQLite.DB(db_path).   It then sets two PRAGMA options for improved write performance:\njournal_mode is set to WAL (Write-Ahead Logging).\nsynchronous is set to NORMAL.\nReturn Value:   The configured SQLite database connection is returned.\n\nArguments\n\ndb_path::String: The file path to the SQLite database. The function will ensure that the directory containing this file exists\nkeep_conn_open::Bool: Optional, whether the connection should persist on the session after the function returns\n\nReturns\n\nAn instance of SQLite.DB representing the open and configured database connection.\n\nExample\n\n```julia db = opendb(\"data/mydatabase.sqlite\", keepconn_open=\"true\")\n\nUse the database connection...\n\nSQLite.execute(db, \"SELECT * FROM my_table;\")\n\nDon't forget to close the database when done.\n\nclose_db(db)\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.random_embeddings-Tuple{SQLite.DB, Int64}","page":"Home","title":"WunDeeDB.random_embeddings","text":"Randomly retrieve a specified number of embeddings from the SQLite database.\n\nThis function is overloaded to support two usage patterns:\n\nrandom_embeddings(db::SQLite.DB, num::Int): Retrieves embeddings using an active database connection.\nrandom_embeddings(db_path::String, num::Int): Opens the database at the specified path, retrieves embeddings, and then closes the connection.\n\nArguments\n\ndb::SQLite.DB or db_path::String: Either an active SQLite database connection or the file path to the SQLite database.\nnum::Int: The number of random embeddings to retrieve.\n\nReturns\n\nA Dict{String, Any} mapping each embedding's ID (as a string) to its embedding vector.\n\nExample\n\n```julia embeddings = randomembeddings(\"mydatabase.db\", 5) for (id, emb) in embeddings     println(\"ID: 'id', Embedding: \", emb) end\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.supported_distance_metrics-Tuple{}","page":"Home","title":"WunDeeDB.supported_distance_metrics","text":"supported_distance_metrics()\n\nReturns a list of the currently supported distance metrics, such as [\"euclidean\", \"cosine\"].\n\n\n\n\n\n","category":"method"},{"location":"#WunDeeDB.update_description","page":"Home","title":"WunDeeDB.update_description","text":"Update the description in the metadata table.\n\nThis function can be called in two ways:\n\nWith an open SQLite.DB connection.\nWith a database file path, which will open the connection if needed.\n\nIt executes a parameterized query to update the description field. If an error occurs, the function closes the database connection, resets global connection variables, and returns an error message.\n\nArguments:\n\ndb::SQLite.DB or db_path::String: A SQLite database connection or the path to the database file.\ndescription::String (optional): The new description to set (defaults to an empty string).\n\nReturns:\n\ntrue on success, or a string with the error message on failure\n\n\n\n\n\n","category":"function"},{"location":"#WunDeeDB.update_embeddings-Tuple{SQLite.DB, Any, Any}","page":"Home","title":"WunDeeDB.update_embeddings","text":"Update one or more embeddings in the SQLite database with new embedding data.\n\nThis function is overloaded to support two usage patterns:\n\nupdate_embeddings(db::SQLite.DB, id_input, new_embedding_input): Updates embeddings using an active database connection.\nupdate_embeddings(db_path::String, id_input, new_embedding_input): Opens the database at the specified path, updates the embeddings, and then closes the connection.\n\nThe function accepts a single identifier or an array of identifiers along with corresponding new embedding vectors. It validates that all new embeddings have the same length, and that their length and data type match the values stored in the meta table. For single record updates, it additionally confirms that the record exists in the database.\n\nArguments\n\ndb::SQLite.DB or db_path::String: Either an active database connection or the file path to the SQLite database.\nid_input: A single ID or an array of IDs identifying the embeddings to update.\nnew_embedding_input: A single numeric embedding vector or an array of such vectors. All embeddings must be of consistent length.\n\nReturns\n\ntrue if the update is successful.\nA String error message if an error occurs.\n\nExamples\n\nUsing an active database connection: ```julia result = update_embeddings(db, 1, [0.5, 0.6, 0.7]) if result === true     println(\"Embedding updated successfully.\") else     println(\"Error: \", result) end\n\nUsing a database file path:\n\nresult = updateembeddings(\"mydatabase.db\", [1, 2], [[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]]) if result === true     println(\"Embeddings updated successfully.\") else     println(\"Error: \", result) end\n\n\n\n\n\n","category":"method"}]
}

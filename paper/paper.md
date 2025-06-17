---
title: "WunDeeDB.jl: An easy to use, zero config, WAL, SQLite backend vector database"
tags:
  - Julia
  - ANN
  - Embeddings
  - Search
  - Vector DB
authors:
  - name: "Alexander V. Mantzaris"
    orcid: 0000-0002-0026-5725
    affiliation: 1
affiliations:
  - name: "Department of Statistics and Data Science, University of Central Florida (UCF), USA"
    index: 1
date: March 21 2025
bibliography: paper.bib
---


# Summary

WunDeeDB.jl is a package written in Julia [@julia] that provides a disk-backed system for storing, searching, and managing embedding vectors at scale, influenced by disk-oriented graph-based ANN techniques [@pan2023lm; @jayaram2019diskann; @singh2021freshdiskann] and the broader insights from hierarchical small-world graphs [@malkov2018efficient; @wang2021comprehensive]. By maintaining embeddings in an SQLite database and optionally using graph-based indices, WunDeeDB.jl minimizes in-memory overhead while supporting efficient similarity searches on commodity hardware. Its design also facilitates integration with common vector-database or ML pipelines that rely on embedding retrieval. The widely known DiskANN algorithm, has an open source code base [@diskann-github] and implements many of the core principles that underlines DiskANN development directions.

In contrast to fully in-memory approaches, WunDeeDB.jl leverages disk-based storage and user-configurable adjacency (e.g., HNSW, LM-DiskANN, or fallback linear search), allowing large-scale data to be handled without saturating RAM. It supports incremental insertions and deletions, ensuring the index remains up-to-date as datasets evolve. By combining these disk-native strategies with tunable BFS expansions and adjacency pruning, WunDeeDB.jl enables robust nearest neighbor searches for high-dimensional embeddings.

Features include:

- **SQLite-backed embeddings** with automatic consistency checks on dimension and data type.
- **Optional ANN indexing** (HNSW, LM-DiskANN) or linear fallback, selectable at DB initialization.
- **Incremental insert/delete** operations that update disk-based adjacency structures to keep pace with dataset changes.
- **Configurable BFS expansions** (e.g., `EF_SEARCH`, `EF_CONSTRUCTION`) to balance search speed vs. recall.
- **Choice of Precision** the user can choose any of the standard base precisions to store the embeddings as.

With these capabilities, **WunDeeDB.jl** offers a practical and scalable solution for disk-based embedding management, building on research showing the viability of disk-native approaches for large ANN indexes [@pan2023lm; @jayaram2019diskann; @singh2021freshdiskann]. It is designed for minimal configuration, providing Write-Ahead Logging (WAL) and transactions. 


# Statement of Need

Approximate Nearest Neighbor (ANN) search is a key element in recommendation systems, large-scale retrieval, and embedding-based machine learning [@wang2021comprehensive]. Traditional in-memory approaches often face significant memory demands and slower scaling when dealing with more points than can fit in core memory. By persisting adjacency structures on disk rather than in RAM, WunDeeDB.jl tackles these bottlenecks—building on disk-based ANN research [@pan2023lm; @jayaram2019diskann; @singh2021freshdiskann] and provides:

1. **Reduced Memory Overhead**: Only a fraction of data resides in memory, making it feasible to handle larger datasets on commodity hardware.  
2. **Dynamic Updates**: Graph-based indexing supports insertions and deletions, allowing adaptation to evolving or streaming data.  
3. **High Recall**: Adjusting BFS expansions and adjacency parameters yields near state-of-the-art accuracy in neighbor retrieval.  
4. **Scalable & Simple Architecture**: Built using Julia’s performance ecosystem, WunDeeDB.jl integrates disk operations with numeric libraries, and is straightforward to install.

This approach benefits practitioners needing large-scale nearest neighbor indices without requiring specialized clusters or massive RAM. Within the Julia package ecosystem such an implementation is not currently provided. The closest package is introduced in @tellez2022similaritysearch, which provides essential ANN algorithms but they reside in memory and not disk, do not provide data protection through transactions or journaling such as WAL (Write Ahead Logging). For users that need these features along with ANN, this package provides them as well as being cross platform.


# Acknowledgements

Thanks to the Julia community for their continued support of open-source scientific computing.

# References
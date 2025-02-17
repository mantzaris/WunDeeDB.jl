```@meta
CurrentModule = WunDeeDB
```

```@contents

```


# WunDeeDB

Documentation for [WunDeeDB](https://github.com/mantzaris/WunDeeDB.jl).

## Introduction

This module supports bulk operations (insertions, deletions, and updates) with a hard limit of 1000 records per operation. In addition, the module supports the following numerical types for embedding data:

-    Float16
-    Float32
-    Float64
-    BigFloat
-    Int8
-    UInt8
-    Int16
-    UInt16
-    Int32
-    UInt32
-    Int64
-    UInt64
-    Int128
-    UInt128

These types are defined in the DATA_TYPE_MAP and are used to correctly parse and manage the embedding vectors stored in the SQLite database. The constant BULK_LIMIT is set to 1000 to prevent overly large transactions during bulk operations.


```@index
```

```@autodocs
Modules = [WunDeeDB]
Private = false
Order = [:function]
```
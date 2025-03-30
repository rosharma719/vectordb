# VectorDB: High-Performance In-Memory Vector Search Engine

## Overview

VectorDB (better name coming soon!) is a lightweight, high-performance in-memory vector search engine designed for efficient similarity search and payload storage. 

## Features

### Current Capabilities
- **HNSW (Hierarchical Navigable Small World) Indexing**
  - Optimized approximate nearest neighbor search
  - Supports arbitrarily high-dimensional vector spaces
  - Efficient search with configurable trade-offs between accuracy and performance (user-friendly interface in the works)
  - Supports common distance metrics (Cosine, Euclidean, Dot)

- **Payload Storage**
  - Store additional metadata alongside vector embeddings (ints, floats, strings, homogeneous lists)
  - Seamless integration with vector search operations

## Roadmap

- [x] HNSW Indexing
- [x] Payload Storage
- [x] Inverse Indexing
- [x] Vector Deletion
- [x] In-Place Filtering
- [X] Filtering and Query Schema
- [ ] Python API

## Other potential features

- [ ] Persistence
- [ ] Mutable/immutable segmentation
  - [ ] Compression and quantization for fast immutable segment search
- [ ] Graph functionality
- [ ] Generative AI query builder

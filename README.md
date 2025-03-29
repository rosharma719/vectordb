# VectorDB: High-Performance In-Memory Vector Search Engine

## Overview

VectorDB (better name coming soon!) is a lightweight, high-performance in-memory vector search engine designed for efficient similarity search and payload storage. Built with performance and flexibility in mind, it provides robust vector indexing and retrieval capabilities.

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

### Upcoming Features
- Vector deletion
- In-place filtering with custom query schema
- Python API

## Roadmap

- [x] HNSW Indexing
- [x] Payload Storage
- [x] Inverse Indexing
- [x] Vector Deletion
- [ ] In-Place Filtering
- [ ] Query Schema
- [ ] Python API

# Vector Database Optimization and Configuration
**Objective 6.2: Configure and optimize vector databases for fast retrieval**

Vector databases are the backbone of high-performance RAG systems. This guide covers indexing strategies, configuration tuning, and NVIDIA-accelerated approaches for production-scale knowledge retrieval.

## Indexing Algorithms: When to Use What

### HNSW (Hierarchical Navigable Small World)
**Best for:** General-purpose similarity search with consistent performance requirements

- Approximate nearest neighbor search using hierarchical layer structure
- Fast query speed with excellent recall (95%+)
- Higher memory overhead than alternatives
- Configuration parameters:
  - `ef_construction`: Higher values improve recall at indexing cost (default 200, range 100-400)
  - `M`: Maximum number of connections per node (default 16, range 5-48)
    - Lower M: faster indexing, lower memory
    - Higher M: better search quality but slower indexing
  - `ef`: Query-time parameter (higher = slower but better recall)

```python
# HNSW Configuration Example
import milvus

# Milvus with HNSW
hnsw_params = {
    "index_type": "HNSW",
    "metric_type": "L2",  # or "IP" (inner product)
    "params": {
        "M": 32,              # Dense connections
        "efConstruction": 360 # High-quality indexing
    }
}

collection.create_index(
    field_name="embeddings",
    index_params=hnsw_params
)
```

**Trade-offs:**
- Query latency: 5-20ms (single vector)
- Memory per vector: ~1.5-2x embedding size
- Index build time: Moderate (slow for billions of vectors)

### IVF-Flat (Inverted File with Flat Quantization)
**Best for:** High-throughput queries with tight memory constraints

- Clusters vectors into `nlist` groups (inverted lists)
- Does exact distance computation within probed clusters
- Lower memory per vector than HNSW
- Configuration parameters:
  - `nlist`: Number of clusters (typical 128-1024)
    - Tradeoff: more clusters = finer granularity but slower probe
  - `nprobe`: Clusters to search at query time (1-nlist)
    - Lower nprobe: faster queries, lower recall
    - Higher nprobe: slower queries, better recall

```python
# IVF-Flat Configuration
ivf_flat_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {
        "nlist": 256,      # 256 clusters
        "nprobe": 16       # Search 16 clusters (6% of total)
    }
}

collection.create_index(
    field_name="embeddings",
    index_params=ivf_flat_params
)
```

**Trade-offs:**
- Query latency: 10-50ms
- Memory per vector: 1x embedding size
- Index build time: Very fast
- Recall: 70-85% (depends on nprobe)

### IVF-PQ (Inverted File with Product Quantization)
**Best for:** Billion-scale vectors with memory constraints and offline batch queries

- Combines IVF clustering with Product Quantization (PQ)
- Compresses vectors: 768-dim embeddings to 96-128 bytes
- Best compression-to-recall ratio
- Configuration parameters:
  - `nlist`: Cluster count (typical 1024-4096 for large-scale)
  - `m`: Number of PQ subvectors (typical 8-64, higher = better quality, higher memory)
  - `nbits`: Bits per PQ code (8 typical, 4 for extreme compression)

```python
# IVF-PQ for Billion-Scale Vector Storage
ivf_pq_params = {
    "index_type": "IVF_PQ",
    "metric_type": "L2",
    "params": {
        "nlist": 2048,     # 2048 clusters for 1B vectors
        "m": 8,            # 8 subvectors
        "nbits": 8         # 8 bits per code
    }
}

# Typically used with SQ8 post-processing for fine-tuning
```

**Trade-offs:**
- Query latency: 50-200ms (acceptable for batch processing)
- Memory per vector: 0.1x embedding size (dramatic compression)
- Index build time: Slow but one-time cost
- Recall: 60-75% (compression loss)

### Brute Force (Exact Search)
**Best for:** Small datasets (<10M vectors), exact recall requirements, benchmarking

- Linear scan, distance computed for every vector
- 100% recall guaranteed
- No indexing overhead

```python
brute_force_params = {
    "index_type": "FLAT",
    "metric_type": "L2"
}
```

**Trade-offs:**
- Query latency: O(n) = milliseconds to seconds
- Memory: Minimal overhead
- Use only for ground truth or small corpora

## Algorithm Selection Decision Tree

```
Do you have < 100K vectors?
├─ YES: Use HNSW (speed and simplicity)
└─ NO: Is memory a primary constraint?
   ├─ YES: Is latency critical (<50ms)?
   │  ├─ YES: Use IVF-PQ with small nbits
   │  └─ NO: Use IVF-PQ standard
   └─ NO: Use HNSW (best all-around)
```

## GPU-Accelerated Search

### Milvus with GPU Support
Milvus supports GPU acceleration for index building and search using NVIDIA CUDA.

```python
from milvus import connections, Collection

# Connect to Milvus with GPU
connections.connect(
    alias="default",
    host="localhost",
    port=19530,
    pool_size=10
)

# Create collection
collection = Collection("large_embeddings")

# GPU-accelerated HNSW indexing
gpu_hnsw = {
    "index_type": "GPU_HNSW",  # GPU variant
    "metric_type": "L2",
    "params": {
        "M": 32,
        "efConstruction": 256
    }
}
collection.create_index(
    field_name="embeddings",
    index_params=gpu_hnsw
)

# Query with GPU-accelerated search
results = collection.search(
    data=[query_embedding],
    anns_field="embeddings",
    param={"metric_type": "L2", "params": {"ef": 64}},
    limit=10,
    expr="metadata > 0"
)
```

**Performance Gains:**
- Index building: 3-5x faster on GPU vs CPU
- Query throughput: 10-100x improvement for batch queries
- Latency: 1-5ms per query (vs 20-50ms on CPU)

### FAISS with GPU Acceleration
Facebook AI Similarity Search (FAISS) provides optimized GPU kernels.

```python
import faiss
import numpy as np

# Create GPU index
def create_gpu_index(vectors, index_type="HNSW"):
    """Create FAISS index on GPU"""
    d = vectors.shape[1]  # embedding dimension

    if index_type == "HNSW":
        cpu_index = faiss.IndexHNSWFlat(d, 32)  # M=32
        cpu_index.hnsw.efConstruction = 256
        gpu_index = faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(),
            0,  # GPU device ID
            cpu_index
        )
    elif index_type == "IVF":
        # IVF on GPU (very fast)
        index = faiss.GpuIndexIVFFlat(
            faiss.StandardGpuResources(),
            d,
            nlist=256,
            metric=faiss.METRIC_L2
        )

    gpu_index.add(vectors)
    return gpu_index

# Batch search on GPU
def gpu_search(index, queries, k=10):
    distances, indices = index.search(queries, k)
    return indices, distances

# Example: 1M vectors, 1K queries
vectors = np.random.random((1_000_000, 768)).astype(np.float32)
queries = np.random.random((1000, 768)).astype(np.float32)

gpu_index = create_gpu_index(vectors, "HNSW")
indices, distances = gpu_search(gpu_index, queries, k=10)
```

**FAISS GPU Benchmarks:**
- 1M vectors, 768-dim: Index build 2-3 seconds
- Batch query (1K queries): 50-100ms
- Single query: 1-2ms

## Vector Database Comparison

| Feature | Milvus | Pinecone | Weaviate | Qdrant | Chroma |
|---------|--------|----------|----------|--------|--------|
| **Scale** | 100B+ | 100B+ | 10B | 10B | 100M |
| **GPU Support** | Yes (native) | Yes (backend) | No | No | No |
| **Self-Hosted** | Yes | No | Yes | Yes | Yes |
| **License** | Open (AGPL) | Proprietary | Open (AGPL) | Open (AGPL) | Open (Apache) |
| **Cost (1B vectors)** | $100s | $1000s/mo | $100s | $100s | Free |
| **Query Latency** | 5-10ms | 50-200ms | 20-50ms | 10-20ms | 1-10ms |
| **Metadata Filtering** | Excellent | Good | Excellent | Good | Basic |
| **Replication** | Built-in | Managed | Built-in | Built-in | File-based |
| **Best For** | Enterprise RAG | Managed service | Knowledge graphs | Real-time | Dev/testing |

## Configuration Tuning for Performance

### Milvus Performance Tuning

```yaml
# milvus.yaml configuration
common:
  logLevel: warning

queryCoord:
  autoHandoff: true
  autoBalance: true
  balancerName: rowCountBalancer

queryNode:
  cache:
    enable: true
    memoryLimit: 3GB  # Cache for hot vectors

indexCoord:
  indexBuildParallel: 4  # Parallel index builds

# GPU configuration
gpu:
  index:
    device_id: 0
    enable: true
    memory_fraction: 0.7  # Use 70% of GPU memory
```

### Index Selection by Scenario

**Scenario 1: Real-time Finance Data (sub-10ms latency)**
```
Index: HNSW
Configuration:
- ef_construction: 256
- M: 24
- ef: 32
- GPU: Required
Reason: Consistent low latency with good recall
```

**Scenario 2: Enterprise Document RAG (50-100K docs)**
```
Index: IVF-Flat
Configuration:
- nlist: 256
- nprobe: 32
- Memory: <10GB
- GPU: Optional
Reason: Fast indexing, simple tuning, sufficient recall
```

**Scenario 3: Billion-Scale Research Archive**
```
Index: IVF-PQ
Configuration:
- nlist: 4096
- m: 8, nbits: 8
- Sharding: 10 shards
- GPU: Batch indexing
Reason: Memory efficiency, acceptable batch latency
```

## Scaling Strategies

### Horizontal Scaling with Sharding
```python
# Milvus sharding example
collection.create_index(
    field_name="embeddings",
    index_params={
        "index_type": "HNSW",
        "params": {"M": 32, "efConstruction": 256}
    }
)

# Set shard count
collection.load(
    replica_number=3,  # 3 replicas for HA
)

# Shard key for distribution
partition_keys = ["user_id", "document_source"]
```

**Scaling Benefits:**
- Distributes vectors across multiple nodes
- Parallelizes indexing and search
- Linear throughput scaling to 10K+ QPS

### Replication and High Availability
```python
# Milvus replication setup
replica_number = 3
collection.load(replica_number=replica_number)

# Query with load balancing
search_params = {
    "metric_type": "L2",
    "params": {"ef": 64},
    "round_robin": True  # Load balance across replicas
}
```

## Performance Trade-Off Matrix

```
High Recall (>95%) + Low Latency (<10ms)
├─ HNSW on GPU with ef=256+
├─ Cost: 2-3x memory of raw vectors
└─ Use case: Real-time trading, medical search

Good Recall (85-90%) + Medium Latency (10-50ms)
├─ HNSW on CPU or IVF-Flat on GPU
├─ Cost: 1-1.5x memory
└─ Use case: Most production RAG systems

Acceptable Recall (70-80%) + High Latency (50-500ms)
├─ IVF-PQ or IVF-Flat with low nprobe
├─ Cost: 0.1-0.5x memory
└─ Use case: Batch processing, archive search

Exact Recall (100%) + Slow Query (seconds)
├─ Brute force + distributed compute
├─ Cost: Minimal storage, maximal compute
└─ Use case: Ground truth validation, quality assurance
```

## NVIDIA Integration Points

**NVIDIA NIM for Vector Search:**
- Deploy optimized vector search microservices
- Leverages NVIDIA RAPIDS for GPU-accelerated ETL
- Integrated with LLMs for end-to-end retrieval

**NVIDIA RAPIDS:**
- GPU-accelerated data preparation before indexing
- CuML for embedding similarity computation
- cuDF for metadata filtering at scale

## Study Questions

**Q1:** You need to index 500 million 768-dimensional embeddings with <20ms query latency and a budget of $5K for infrastructure. Which approach should you choose?

A) Pinecone (managed) - Costs ~$4K/month but handles scale easily
B) Milvus with HNSW on GPU - One-time cost, fast but requires tuning
C) FAISS with IVF-PQ and distributed serving - Requires engineering
D) Qdrant with custom sharding - Good balance of cost and simplicity

**Answer: D** - Qdrant offers self-hosted flexibility with good GPU support and reasonable ops overhead.

---

**Q2:** Your HNSW index shows 92% recall but queries take 45ms. You want <10ms latency. What's the first thing you should try?

A) Reduce ef_construction from 300 to 100
B) Decrease M from 32 to 16
C) Decrease ef from 200 to 32 for queries
D) Switch to IVF-Flat indexing

**Answer: C** - Query-time parameter `ef` directly controls latency/recall tradeoff. Reduce it first before rebuilding.

---

**Q3:** You're storing 2 billion vectors but only have 64GB of GPU memory. Your current HNSW index uses 120GB. What's the best compression strategy?

A) Switch to IVF-Flat with larger nlist
B) Switch to IVF-PQ with m=8, nbits=8
C) Enable NVIDIA RAPIDS cuQuantize for vector quantization
D) Shard across 4 GPUs with Milvus

**Answer: B** - IVF-PQ compresses to ~0.1x original size. With m=8, nbits=8: 2B × 1 byte ≈ 2GB index size.

---

**Q4:** For a financial trading application requiring sub-5ms latency and 99% recall on 100M vectors, what's the optimal setup?

A) GPU HNSW with ef_construction=400, M=40, batch GPU queries
B) Distributed CPU HNSW with aggressive caching
C) IVF-Flat on GPU with nprobe=256
D) Multiple FAISS GPU indices with round-robin balancing

**Answer: A** - Trading needs both speed and accuracy. GPU HNSW with high construction effort (one-time cost) and conservative query parameters meets both SLOs.

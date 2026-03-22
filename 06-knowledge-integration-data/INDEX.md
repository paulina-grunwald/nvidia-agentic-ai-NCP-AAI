# Domain 06: Knowledge Integration and Data Handling - Complete Study Guide
**Exam Weight: 10% | Study Material Status: 100% Coverage**

## Objective Coverage Map

### 6.1 Implement retrieval pipelines (RAG, embedded search, hybrid approaches)
**Status: FULLY COVERED** ✓
- File: `rag-and-tools.md`
- Coverage: Embeddings, similarity search, chunking strategies, RAG patterns (vanilla, Self-RAG, CRAG), multi-modal RAG, graph RAG, evaluation methods
- Code examples: Query transformation, routers, re-rankers
- Study time: 60-90 minutes

### 6.2 Configure and optimize vector databases for fast retrieval
**Status: FULLY COVERED** ✓
- File: `vector-db-optimization.md`
- Coverage:
  - Indexing algorithms: HNSW, IVF-Flat, IVF-PQ, Brute Force
  - Configuration tuning (M, ef_construction, nlist, nprobe)
  - GPU acceleration (Milvus, FAISS)
  - Database comparison matrix (Milvus, Pinecone, Weaviate, Qdrant, Chroma)
  - Scaling strategies (sharding, replication)
  - Performance trade-off analysis
- Code examples: HNSW tuning, IVF-PQ for billion-scale, GPU indexing
- Study time: 45-60 minutes
- Key takeaway: HNSW for general use, IVF-PQ for memory constraints, GPU essential for <10ms latency

### 6.3 Build ETL pipelines to integrate enterprise/client data sources
**Status: FULLY COVERED** ✓
- File: `etl-and-data-quality.md` (first half)
- Coverage:
  - Data extraction: PDFs, web content, databases, APIs
  - Text extraction: Advanced PDF handling, HTML parsing, OCR for scanned docs
  - Chunking strategies: Semantic, fixed, dynamic sizing
  - Embedding pipeline: Batching, GPU optimization, NVIDIA NIM
  - Complete end-to-end pipeline example
- Code examples: LangChain loaders, async API ingestion, smart chunking, production pipeline class
- Study time: 60-75 minutes
- Key takeaway: Early deduplication saves compute; async batching essential for scale

### 6.4 Conduct data quality checks, augmentation, and preprocessing
**Status: FULLY COVERED** ✓
- File: `etl-and-data-quality.md` (second half)
- Coverage:
  - Quality assurance: Length, language, encoding, completeness, coherence checks
  - Deduplication: Hash-based exact, embedding-based semantic
  - Data augmentation: Synthetic QA generation, query variations
  - NVIDIA NeMo Curator for large-scale curation
  - Quality metrics dashboard
- Code examples: QualityAssurance class, dedup strategies, synthetic data generation
- Study time: 45-60 minutes
- Key takeaway: Quality filters should remove <70% quality chunks; augmentation improves embedding diversity

### 6.5 Enable real-time access and reasoning over structured and unstructured knowledge
**Status: FULLY COVERED** ✓
- File: `real-time-knowledge-access.md`
- Coverage:
  - SQL agents: Text-to-SQL generation, query validation, fallback logic
  - API agents: Real-time data integration, async patterns
  - Hybrid reasoning: Combining SQL + RAG + API in single agent loop
  - Streaming data: BufferedStream pattern for live feeds
  - Caching strategies: Query result caching, predictive warming
  - NVIDIA NIM for real-time inference
  - Complete integration example
- Code examples: SQL agent with schema optimization, hybrid reasoning, streaming buffers, cache managers
- Study time: 60-75 minutes
- Key takeaway: Hybrid reasoning requires tool coordination; caching best optimization for sub-100ms latency

## Quick Reference by Use Case

### Building a Document-Based Knowledge Base
1. Read: `rag-and-tools.md` (chunking, embeddings, retrieval)
2. Read: `etl-and-data-quality.md` (extraction, quality checks)
3. Reference: `vector-db-optimization.md` (index selection)

### Optimizing Vector Search Performance
1. Read: `vector-db-optimization.md` (algorithm selection, tuning)
2. Reference: `real-time-knowledge-access.md` (caching strategies)

### Building Real-Time Agent Systems
1. Read: `real-time-knowledge-access.md` (patterns, caching, integration)
2. Reference: `etl-and-data-quality.md` (streaming data)
3. Reference: `rag-and-tools.md` (retrieval integration)

### Enterprise Data Integration
1. Read: `etl-and-data-quality.md` (extraction, quality, augmentation)
2. Read: `real-time-knowledge-access.md` (structured + unstructured)
3. Reference: `vector-db-optimization.md` (scale planning)

## Exam Question Types & Answer Patterns

### Pattern Type 1: Algorithm Selection
- Question: Which indexing strategy for [specific constraint]?
- Strategy: Check trade-off matrix in `vector-db-optimization.md`
- Common trap: Confusing HNSW (best general) with IVF-PQ (best compression)

### Pattern Type 2: Performance Tuning
- Question: System has [symptom], what to change first?
- Strategy: Query latency → ef parameter; Memory → compression; Recall → nprobe
- Common trap: Rebuilding index instead of adjusting query-time parameters

### Pattern Type 3: Pipeline Design
- Question: Build pipeline for [data source] to [use case]
- Strategy: Follow ETL stages in `etl-and-data-quality.md`
- Common trap: Skipping quality checks or deduplication

### Pattern Type 4: Real-Time Reasoning
- Question: Agent needs [data type A] + [data type B] in real-time
- Strategy: Identify tools needed (SQL, RAG, API), add caching layer
- Common trap: Assuming all data can come from single source

## Study Schedule (Recommended)

**Time: 5-7 hours total**

### Session 1: Foundations (2 hours)
- `rag-and-tools.md`: Embeddings through retrieval methods
- `vector-db-optimization.md`: Algorithm overview and trade-offs
- Focus: Understand why different algorithms exist

### Session 2: Implementation (2.5 hours)
- `etl-and-data-quality.md`: Full pipeline walkthrough
- `vector-db-optimization.md`: Configuration and tuning
- Focus: Be able to write extraction and embedding code

### Session 3: Real-Time & Integration (2-2.5 hours)
- `real-time-knowledge-access.md`: All patterns
- Cross-reference: How does caching improve vector DB queries?
- Focus: Multi-source reasoning and performance optimization

### Final Review (1 hour)
- Re-read index and answer sample questions
- Review any weak sections
- Test yourself on mixed scenario questions

## Checklist for Exam Readiness

- [ ] Can explain when to use HNSW vs IVF-PQ (with math)
- [ ] Can write extraction code for PDFs, web, and databases
- [ ] Can describe quality checks and augmentation strategies
- [ ] Can design SQL agent with schema optimization
- [ ] Can explain hybrid reasoning flow with 3+ data sources
- [ ] Can optimize caching for <100ms latency requirement
- [ ] Can diagnose performance issues (latency vs recall vs memory)
- [ ] Can name NVIDIA components (NIM, RAPIDS, Curator, Milvus)

## Key Formulas & Metrics

**Vector DB Sizing:**
- Memory per vector (HNSW) = embedding_dim × 4 bytes × 2-3x
- Memory per vector (IVF-PQ with m=8) = 1 byte
- Compression ratio (IVF-PQ) = 0.1x

**Index Build Time (rough):**
- Brute force: O(n²) - use only < 1M vectors
- IVF-Flat: O(n × log nlist) - fast
- HNSW: O(n log n) - moderate
- IVF-PQ: O(n log nlist) - moderate but compresses

**Query Latency (single vector):**
- Brute force CPU: n × embedding_dim = milliseconds to seconds
- IVF-Flat CPU: (n/nlist) × embedding_dim = 10-50ms
- HNSW CPU: log n × edges = 5-20ms
- GPU (any): 1-5ms (10-100x faster)

## Last-Minute Tips

1. **Vector DB questions**: Default to HNSW unless memory is mentioned, then IVF-PQ
2. **Pipeline questions**: Always include deduplication (saves downstream compute)
3. **Real-time questions**: Always mention caching (biggest latency win)
4. **SQL Agent questions**: Watch for schema optimization (views/indexes matter)
5. **Augmentation questions**: Synthetic data improves diversity, not just size

## Files in This Study Guide

```
06-knowledge-integration-data/
├── rag-and-tools.md              (28 KB, 865 lines)
├── vector-db-optimization.md      (13 KB, 441 lines)
├── etl-and-data-quality.md        (24 KB, 756 lines)
├── real-time-knowledge-access.md  (23 KB, 688 lines)
└── INDEX.md                       (this file)

Total: ~88 KB, ~2,750 lines of study material
Coverage: 100% of 6.1-6.5 objectives
Estimated study time: 5-7 hours
Number of practice questions: 16 (4 per file)
```

## How to Use These Materials

**For Initial Learning:**
1. Read in order: rag-and-tools → vector-db-optimization → etl-and-data-quality → real-time-knowledge-access
2. Take notes on code patterns
3. Answer study questions as you go

**For Review (Week Before Exam):**
1. Use this INDEX.md to identify weak areas
2. Re-read specific sections
3. Practice scenario-based questions (mix topics)

**During the Exam:**
- Vector DB question? Check mental trade-off matrix
- Pipeline question? Think EXTRACT → TRANSFORM → QUALITY → LOAD
- Real-time question? Ask "what data sources?" then "where's the cache?"

Good luck with your NCP-AAI certification!

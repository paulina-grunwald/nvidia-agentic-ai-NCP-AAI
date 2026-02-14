# RAG - Retrieval Augmented Generation

## Overview

- **Tool calling** enabled LLMs to interact with external environments through a dynamic interface where tool calling not only helps choose the appropriate tool but also infer necessary arguments for execution.
- In standard RAG, LLMs are mainly used for synthesis of information only.

## Embeddings

- **Embedding** is a high-dimensional data as vectors in a lower-dimensional space => This matters because once words are vectors, we can mathematically measure how similar they are (using dot product or cosine similarity).
- For example, words like "cat" and "dog" can be represented as vectors that are close to each other in the vector space, reflecting their semantic similarity. This is in contrast to one-hot encoding, where each word is represented as a sparse binary vector, which fails to capture any semantic relationships.
- **One-hot encoding**:
  - one-hot encoding is just an ID system (like assigning each word a number), while dense embeddings actually encode what words mean relative to each other.
  ```
  king  = [1, 0, 0, 0, 0]   ← the 1 is at position 0
  queen = [0, 1, 0, 0, 0]   ← the 1 is at position 1
  (1×0) + (0×1) + (0×0) + (0×0) + (0×0) = 0
  ```
- **Dense embeddings**
- are powerful — they have non-zero values at many shared positions e.g

  ```
  cat  = [0.7, 0.3, -0.1, 0.9]
  dog  = [0.6, 0.4, -0.2, 0.8]
  (0.7×0.6) + (0.3×0.4) + (-0.1×-0.2) + (0.9×0.8) = 1.28 ← similar!

  ```

- Note: in real models, embedding vectors have hundreds of dimensions (e.g., 768 or 1536), not just 4!

### Embedding Models

- Embedding models are machine learning algorithms, often based on transformer architectures, that transform data like text, images, or audio into low-dimensional, dense vectors (numerical arrays).
- Embedding Models - Convert text → numbers (vectors) that capture meaning

| Model Type         | Examples                     | Dimensions | Use Case                        |
| ------------------ | ---------------------------- | ---------- | ------------------------------- |
| **Small/Fast**     | all-MiniLM-L6                | 384        | Quick prototypes, low resources |
| **Balanced**       | e5-base, bge-base            | 768        | Production balance              |
| **Large/Accurate** | e5-large, bge-large          | 1024       | High accuracy needs             |
| **Specialized**    | NeMo Retriever, Cohere embed | Varies     | Enterprise, multilingual        |

# Router Engine

- A Router is a component that decides which **tool**, **index**, or **model** to use based on the incoming query. Think of it as a traffic controller for your agent.
- Without router it would have to look through all documents

```mermaid
                    ┌─────────────────┐
                    │     ROUTER      │
                    │  "Where should  │
     Query ───────► │   this go?"     │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐      ┌──────────-┐
    │  Index A │      │  Index B │      │  Tool C   │
    │ (Finance)│      │ (Legal)  │      │(Calculator)│
    └──────────┘      └──────────┘      └──────────-┘
```

#### Types of Routers

- **LLM Selector Router (Most common)**
  - Uses an LLM to understand the query and pick the best destination.

    | Pros                        | Cons                   |
    | --------------------------- | ---------------------- |
    | Understands complex queries | Adds LLM latency       |
    | Handles ambiguous intent    | Costs money (LLM call) |
    | Can reason about routing    | Can make wrong choices |

- **Embedding Router (Semantic Similarity)**
  - Compares query embedding to description embeddings of each tool/index.
  - Clear-cut categories, speed-critical applications

  ```tsx
  Query: "revenue forecast"
      ↓
  Embed query
      ↓
  Compare to tool descriptions:
  - "Financial data and reports" → 0.89 similarity ✓
  - "Legal contracts" → 0.32 similarity
  - "HR policies" → 0.21 similarity
      ↓
  Route to Finance (highest similarity)
  ```

  | Pros               | Cons                           |
  | ------------------ | ------------------------------ |
  | Fast (no LLM call) | Less nuanced understanding     |
  | Cheap              | Depends on description quality |
  | Deterministic      | Can't handle complex logic     |

- **Pydantic Router (Structured Output)**

  | Pros                       | Cons                   |
  | -------------------------- | ---------------------- |
  | Type-safe, predictable     | Still needs LLM call   |
  | Easy to debug              | Less flexible          |
  | Includes confidence scores | Requires schema design |

  **Best for:** Production systems needing reliability

- **Keyword/Rule-Based Router**

  ```tsx
  def route(query):
      if "revenue" in query or "profit" in query:
          return finance_index
      elif "contract" in query or "legal" in query:
          return legal_index
      else:
          return general_index
  ```

  - The agent uses **ReAct reasoning** to decide routing at each step!

  | Pros        | Cons                    |
  | ----------- | ----------------------- |
  | Fastest     | Misses semantic meaning |
  | No cost     | Requires manual rules   |
  | Predictable | Doesn't scale           |

#### Exam patterns

"Agent needs to choose between multiple data sources" -> Router
"Query should go to appropriate index based on content" -> Router
"Need fast routing without LLM" -> Embedding Router
"Complex routing decisions with reasoning" -> LLM Selector Router
"Type-safe, structured routing" -> Pydantic Router

Router - BEFORE retrieval - Decides WHERE to search
Re-ranker - AFTER retrieval - Reorders WHAT was found

# Similarity_top_k

- Number of most similar chunks to retrieve
- Query: "What is the company revenue?"
  - similarity_top_k=3 → Returns 3 most relevant chunks
  - similarity_top_k=10 → Returns 10 most relevant chunks
    Tradeoffs:
    | Low top_k (1-3) | High top_k (10+) |
    | --- | --- |
    | Faster | Slower |
    | More focused | More comprehensive |
    | May miss context | May include noise |
    | Lower cost | Higher cost (more tokens) |

- 🔥 **Exam Tip:** If question mentions "**missing relevant context**" → increase top_k

### Standard RAG vs Agentic RAG

| Aspect      | Standard RAG                | Agentic RAG                                      |
| ----------- | --------------------------- | ------------------------------------------------ |
| Flow        | Query → Retrieve → Generate | Query → _Reason_ → _Tools_ → Retrieve → Generate |
| LLM Role    | Synthesis only              | **_Reasoning + Tool selection_**                 |
| Flexibility | Fixed pipeline              | Dynamic decisions                                |
| Tools       | Just retriever              | Multiple tools (search, calculate, API)          |
| Multi-step  | No                          | Yes (reasoning loop)                             |

- 🔥 Exam Pattern: "Agent needs to decide WHAT to retrieve and HOW" → Agentic RAG

### RAG Pipeline Components

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │ Documents│ → │ Chunking │ → │ Embedding│ → │ Vector   │  │
│  │ (Input)  │   │          │   │ Model    │   │ Store    │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│                                                             │
│                         INDEXING                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │ Query    │ → │ Retrieval│ → │ Re-rank  │ → │ Generate │  │
│  │          │   │          │   │(optional)│   │ (LLM)    │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│                                                             │
│                         QUERYING                            │
└─────────────────────────────────────────────────────────────┘
```

### Chunking Strategies

| Strategy       | Description                                | Best For             |
| -------------- | ------------------------------------------ | -------------------- |
| **Fixed Size** | Split by character count (e.g., 512 chars) | Simple, fast         |
| **Sentence**   | Split at sentence boundaries               | Maintaining meaning  |
| **Paragraph**  | Split at paragraph breaks                  | Structured docs      |
| **Semantic**   | Split by meaning/topic                     | Best quality, slower |
| **Recursive**  | Try multiple splitters                     | General purpose      |

- **Key Parameters:**
  - `chunk_size`: Size of each chunk (tokens or chars)
  - `chunk_overlap`: Overlap between chunks (prevents cutting context)

* Exam Pattern: "Context is getting cut off" → Increase chunk_overlap

### Metadata Filtering

- Filter chunks BEFORE semantic search:

```
# Example: Only search documents from 2024
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="year", value="2024"),
        MetadataFilter(key="department", value="engineering")
    ]
)
```
Benefits:
- Reduces search space
- More relevant results
- Faster queries
- Enables access control

🔥 Exam Pattern: "Agent should only access certain documents based on user role" → Metadata filtering

### Retrieval Methods

| Method | How It Works | Pros | Cons |
| --- | --- | --- | --- |
| **Dense (Vector)** | Embedding similarity | Semantic understanding | Misses exact matches |
| **Sparse (BM25/TF-IDF)** | Keyword matching | Good for exact terms | No semantic understanding |
| **Hybrid** | Combines both | Best of both worlds | More complex |

**Hybrid Search Formula:**
```javascript
final_score = α × dense_score + (1-α) × sparse_score
```

🔥 **Exam Pattern:** "User searches for exact product code but also wants similar items" → Hybrid search

### Hybrid Search (Lexical + Semantic)

**The Problem**: Pure vector search can miss exact keyword matches; pure keyword search misses semantic meaning.

**The Solution**: Combine both approaches and rerank results.

```javascript
lexical_results = bm25_search(query, top_k=20)      # Keyword matching
semantic_results = vector_search(query, top_k=20)   # Embedding similarity
combined = rerank(lexical_results + semantic_results, top_k=10)
```

**Why it works:**
- Lexical search catches: product codes, exact phrases, acronyms (e.g., "RTX 4090")
- Semantic search catches: paraphrased questions, synonyms, conceptual matches
- ***Reranking*** eliminates duplicates and orders by relevance

**NVIDIA Implementation:**
- **NeMo Retriever** supports hybrid search natively
- Uses **TensorRT-LLM** for fast embedding generation
- Integrates with **Milvus** vector database for scalable search

### Re-ranking

Re-rank retrieved results for better relevance:

```javascript
Initial retrieval (top_k=20) → Re-ranker → Final results (top_n=5)
```

**Re-ranking Models:**
- Cross-encoders (more accurate, slower)
- Cohere Rerank
- NVIDIA NeMo Retriever Re-ranker

**When to use:**
- Initial retrieval returns many results
- Need higher precision
- Quality > Speed

### Agent Reasoning Loop (ReAct)

```javascript
┌─────────────────────────────────────┐
│           AGENT LOOP                 │
│                                      │
│  Query → Thought → Action → Observe  │
│            ↑                    │    │
│            └────────────────────┘    │
│         (repeat until done)          │
└─────────────────────────────────────┘
```

**FunctionCallingAgentWorker:**
- Implements the ReAct loop
- Manages tool selection
- Handles multi-step reasoning
- Maintains conversation state

### NVIDIA-Specific RAG Components

| NVIDIA Component | RAG Role |
| --- | --- |
| **AIQ Toolkit** | Enterprise connectors (Confluence, SharePoint), memory providers |
| **NeMo Retriever** | Embedding models, re-rankers |
| **NIM** | Deploy embedding models at scale |
| **Triton** | Serve retrieval models efficiently |
| **RAPIDS cuGraph** | Knowledge graph integration |

🔥 **Exam Pattern:** "Enterprise RAG with existing knowledge bases" → AIQ Toolkit connectors

### RAG vs Fine-tuning

| Aspect | RAG | Fine-tuning |
| --- | --- | --- |
| **Knowledge update** | Easy (update docs) | Hard (retrain) |
| **Cost** | Lower | Higher |
| **Hallucination** | Reduced (grounded) | Can still hallucinate |
| **Latency** | Higher (retrieval step) | Lower |
| **Best for** | Dynamic knowledge | Style/behavior changes |

🔥 **Exam Pattern:** "Knowledge changes frequently" → RAG
🔥 **Exam Pattern:** "Need consistent response style" → Fine-tuning

### Common RAG Failures & Fixes

| Problem | Symptom | Fix |
| --- | --- | --- |
| **Bad chunking** | Context cut off | Adjust chunk_size/overlap |
| **Low top_k** | Missing relevant info | Increase similarity_top_k |
| **No re-ranking** | Irrelevant results at top | Add re-ranker |
| **Embedding mismatch** | Semantically wrong results | Better embedding model |
| **Stale index** | Outdated answers | Re-index regularly |
| **No metadata** | Can't filter results | Add metadata to chunks |

### Exam Question Patterns for RAG

| Question Type | Look For | Answer Points To |
| --- | --- | --- |
| "Missing context" | Retrieval issue | top_k, chunking |
| "Slow retrieval" | Performance | Caching, index optimization |
| "Wrong documents" | Relevance | Re-ranking, hybrid search |
| "Access control" | Security | Metadata filtering |
| "Outdated answers" | Freshness | Re-indexing, sync schedules |
| "Enterprise sources" | Integration | AIQ Toolkit connectors |

### Chunking - The Foundation

| If you see... | The chunking problem is... | Fix |
| --- | --- | --- |
| "Context cut off" | No overlap | Add `chunk_overlap` |
| "Answers incomplete" | Chunks too small | Increase `chunk_size` |
| "Irrelevant content mixed in" | Chunks too big | Decrease `chunk_size` |
| "Related info separated" | Not semantic | Use semantic chunking |

### Embedding Model Selection

**What they do:** Convert text → vectors (numbers) that capture semantic meaning

#### Embedding Model Types

| Model Type | Examples | Dimensions | Use Case |
| --- | --- | --- | --- |
| **Small/Fast** | all-MiniLM-L6 | 384 | Quick prototypes, low resources |
| **Balanced** | e5-base, bge-base | 768 | Production balance |
| **Large/Accurate** | e5-large, bge-large | 1024 | High accuracy needs |
| **NVIDIA** | NeMo Retriever Embeddings | Varies | Enterprise, optimized |

#### Critical Rule

```javascript
❗ SAME embedding model for INDEXING and QUERYING!

❌ Wrong:
Index with: e5-large (1024 dims)
Query with: MiniLM (384 dims)
→ Dimensions don't match, search fails!

✅ Correct:
Index with: e5-large
Query with: e5-large (same model)
```

🔥 **Exam Pattern:** "Search returns irrelevant results after model update" → Embedding mismatch

### Vector Databases & Indexing Algorithms

#### Popular Vector Stores

| Vector DB | Type | Best For |
| --- | --- | --- |
| **FAISS** | Library | Local, research, fast |
| **Milvus** | Database | Production, scalable, NVIDIA partnership |
| **Pinecone** | Managed | Easy setup, serverless |
| **Weaviate** | Database | Hybrid search built-in |
| **Chroma** | Library | Lightweight, prototypes |
| **Qdrant** | Database | Filtering, payloads |

#### Indexing Algorithms (IMPORTANT!)

| Algorithm | How It Works | Accuracy | Speed | Use Case |
| --- | --- | --- | --- | --- |
| **Flat (Brute Force)** | Compare to EVERY vector | 100% | Very slow | <10K vectors |
| **HNSW** | Graph-based approximate | ~95-99% | Very fast | Most production ⭐ |
| **IVF** | Cluster-based search | ~90-95% | Fast | Billion-scale |
| **PQ** | Compressed vectors | Lower | Fast | Memory-constrained |

🔥 **Exam Pattern:** "Fast search on millions of vectors" → HNSW or IVF

### Advanced Chunking Patterns

#### Sentence Window Retrieval

```javascript
Concept: Index SMALL, Return BIG

Index: Small chunks (sentences) for precise matching
Return: Larger window around matched sentence

Document: "... [s1] [s2] [MATCH] [s4] [s5] ..."
                      ↑
Search finds this sentence (precise)
                      ↓
Returns: [s1] [s2] [MATCH] [s4] [s5] (full context)
```

**Benefit:** Precise retrieval + full context!

#### Parent-Child Retrieval

```javascript
┌───────────────────────────────────────┐
│           PARENT CHUNK                │
│  (Large - stored for context)         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│  │ Child 1 │ │ Child 2 │ │ Child 3 │ │
│  │(indexed)│ │(indexed)│ │(indexed)│ │
│  └─────────┘ └─────────┘ └─────────┘ │
└───────────────────────────────────────┘

Search: Matches Child 2 (small, precise)
Return: Parent chunk (large, contextual)
```

**Exam Pattern:** "Precise matching but need full context" → Parent-child or sentence window

#### Hierarchical Chunking

```javascript
Document
  └── Section 1
        └── Paragraph 1.1
        └── Paragraph 1.2
  └── Section 2
        └── Paragraph 2.1
```

Maintains document structure for navigation.

### Query Transformation Techniques

#### HyDE (Hypothetical Document Embeddings)

```javascript
Problem: Query is short, documents are long
"What is RAG?" vs long document explaining RAG

Solution:
Query: "What is RAG?"
    ↓
LLM generates hypothetical answer:
"RAG (Retrieval Augmented Generation) is a technique
that combines retrieval systems with language models..."
    ↓
Embed THIS hypothetical document
    ↓
Search (query now matches document style!)
```

**When to use:** Short queries, document-style answers expected

#### Multi-Query / Query Expansion

```javascript
Original: "RAG best practices"
    ↓
Generate variations:
- "How to implement RAG effectively"
- "RAG optimization techniques"
- "Common RAG patterns and anti-patterns"
    ↓
Search with ALL queries
    ↓
Combine results
```

**When to use:** Ambiguous queries, comprehensive search needed

#### Query Decomposition

```javascript
Complex: "Compare RAG vs fine-tuning for customer service"
    ↓
Decompose:
- "What are advantages of RAG?"
- "What are advantages of fine-tuning?"
- "Customer service chatbot requirements"
    ↓
Search each, combine answers
```

🔥 **Exam Pattern:** "Complex multi-part questions failing" → Query decomposition

### Adaptive RAG

**Adaptive RAG** is an advanced pattern that **dynamically selects different RAG strategies based on query complexity or type**. Think of it as a "smart router" that doesn't just route to different data sources, but to entirely **different processing pipelines**.

```typescript
User Query
    ↓
┌─────────────────────────┐
│   Complexity Classifier │  ← LLM or rule-based classifier
│   (analyzes the query)  │
└─────────────────────────┘
    ↓
┌───────────┬───────────┬───────────┐
│  SIMPLE   │  MODERATE │  COMPLEX  │
│           │           │           │
│ Direct    │ Standard  │ Multi-hop │
│ LLM call  │ RAG       │ RAG +     │
│ (no RAG)  │ pipeline  │ reasoning │
└───────────┴───────────┴───────────┘
```

### Multi-Query vs Adaptive RAG

| Pattern | What It Does | Key Trigger |
| --- | --- | --- |
| **Multi-Query** | Generates multiple variations of SAME query, retrieves for ALL, unions results | "Rephrase", "multiple perspectives", "same question different ways" |
| **Adaptive RAG** | Routes to DIFFERENT strategies based on complexity | "Simple vs complex", "different approaches", "complexity-based routing" |

**Memory trick:**
- Multi-Query = **M**ultiple versions of same question
- Adaptive RAG = **A**djusts strategy based on complexity

### RAG Fusion (Reciprocal Rank Fusion)

```javascript
Query → Generate multiple query variations
         ↓
    ┌────┴────┐
    ↓         ↓         ↓
  Query 1   Query 2   Query 3
    ↓         ↓         ↓
  Results   Results   Results
    ↓         ↓         ↓
    └────┬────┘
         ↓
   Reciprocal Rank Fusion (RRF)
         ↓
   Combined ranked results
```

**RRF Formula:**
```javascript
RRF_score = Σ (1 / (k + rank_i))
where k = constant (usually 60)
```

**Why it works:** Documents ranked highly by multiple queries get boosted

### Self-RAG & CRAG Patterns

#### Self-RAG (Self-Reflective RAG)

Agent critiques its own retrieval and generation:

```javascript
Query → Retrieve → Generate → Self-Critique
                                    ↓
                        "Is this grounded?"
                        "Is retrieval needed?"
                                    ↓
                         Refine if needed
```

**Key self-assessment tokens:**
- `[Retrieve]` - Do I need to retrieve?
- `[IsRel]` - Is retrieved doc relevant?
- `[IsSup]` - Is response supported by docs?
- `[IsUse]` - Is response useful?

#### CRAG (Corrective RAG)

```javascript
Query → Retrieve → Evaluate Relevance
                        ↓
         ┌──────────────┼──────────────┐
         ↓              ↓              ↓
      Correct       Ambiguous       Incorrect
         ↓              ↓              ↓
      Use docs    Refine search    Web search
                                   (fallback)
```

**Exam Pattern:** "Agent should verify retrieval quality" → Self-RAG or CRAG

### The "Lost in the Middle" Problem

**Problem:** LLMs attend more to START and END, less to MIDDLE of context

```javascript
Context: [Doc1] [Doc2] [Doc3] [Doc4] [Doc5] [Doc6]
           ↑                              ↑
        Attended                       Attended
                    ↑
               Less attended!
```

**Solutions:**

| Solution | How |
| --- | --- |
| **Reorder results** | Put most relevant at start AND end |
| **Reduce context** | Only include top 3-5, not 10+ |
| **Summarize middle** | Compress middle docs |
| **Use re-ranker** | Ensure best docs are first |

**Exam Pattern:** "Important info in middle being ignored" → Lost in middle problem

### RAG Evaluation Metrics

#### Retrieval Metrics

| Metric | What It Measures | Interpretation |
| --- | --- | --- |
| **Precision@K** | Relevant in top K / K | How many retrieved are useful |
| **Recall@K** | Relevant found / Total relevant | Did we find all useful docs |
| **MRR** | 1/rank of first relevant | How quickly we find relevant |
| **NDCG** | Normalized ranking quality | Are better docs ranked higher |

#### Generation Metrics

| Metric | What It Measures |
| --- | --- |
| **Faithfulness** | Is answer grounded in retrieved docs? |
| **Answer Relevance** | Does answer address the question? |
| **Context Relevance** | Are retrieved docs relevant to query? |

#### RAGAS Framework

```javascript
RAGAS evaluates:
1. Faithfulness - Answer supported by context?
2. Answer Relevance - Answer addresses question?
3. Context Precision - Retrieved docs relevant?
4. Context Recall - All needed info retrieved?
```

**Exam Pattern:** "Evaluate RAG system quality" → Faithfulness + Context Relevance + Answer Relevance

### Multi-Modal RAG

#### Handling Different Content Types

| Content Type | Approach |
| --- | --- |
| **Text** | Standard text embeddings |
| **Images** | CLIP or vision model embeddings |
| **Tables** | Serialize to text OR keep structured |
| **PDFs** | OCR + layout analysis → chunks |
| **Code** | Code-specific embeddings |

#### Table Handling Strategies

| Strategy | How | Best For |
| --- | --- | --- |
| **Serialize** | Convert table to text | Simple tables |
| **Structured** | Keep as JSON/rows | Query-able tables |
| **Summary** | LLM describes table | Context understanding |
| **Hybrid** | Store both text + structure | Flexible queries |

### Graph RAG (Knowledge Graphs + RAG)

```javascript
Traditional RAG:
Query → Vector search → Chunks → Answer

Graph RAG:
Query → Extract entities → Graph traversal → Related entities
                                ↓
                        Vector search on subgraph
                                ↓
                        Richer, connected context
```

**When to use Graph RAG:**
- Multi-hop reasoning ("Who is the CEO's boss?")
- Relationship queries ("What products does Company X make?")
- Interconnected knowledge

**NVIDIA component:** RAPIDS cuGraph for graph processing

### Production RAG Considerations

#### Caching Strategies

| Cache Type | How It Works |
| --- | --- |
| **Query Cache** | Same exact query → cached results |
| **Semantic Cache** | Similar queries → cached results |
| **Embedding Cache** | Skip embedding computation for known text |

#### Latency Optimization

| Technique | Impact |
| --- | --- |
| **Embedding caching** | Skip embedding computation |
| **Result caching** | Skip retrieval entirely |
| **Async retrieval** | Parallel processing |
| **Smaller top_k** | Less to process |
| **Approximate search (HNSW)** | vs brute force |
| **Metadata filtering** | Smaller search space |

#### Cost Management

```javascript
Costs in RAG:
1. Embedding API calls (per token)
2. Vector DB storage (per vector)
3. LLM API calls (per token in context)

Reduce costs:
- Cache embeddings
- Reduce chunk overlap (carefully)
- Lower top_k
- Compress context before LLM
```

### RAG Anti-Patterns (What NOT to Do)

| Anti-Pattern | Problem | Fix |
| --- | --- | --- |
| **Different embedding models** | Dimension mismatch | Same model for index & query |
| **No chunk overlap** | Lost context at boundaries | Add 10-20% overlap |
| **Too many results to LLM** | Context overflow, noise | Re-rank, lower top_k |
| **No re-indexing schedule** | Stale answers | Automated re-indexing |
| **Ignoring metadata** | Can't filter or scope | Add metadata during indexing |
| **Flat file storage** | No relationships | Use proper vector DB |

### Document Lifecycle in RAG (CRITICAL!)

```javascript
┌─────────────────────────────────────────────────────┐
│              DOCUMENT LIFECYCLE                      │
│                                                       │
│  Upload → Parse → Chunk → Embed → Index → SEARCHABLE  │
│    📄      🔍      ✂️       🔢      🗄️        ✅          │
│                                                       │
│  ⚠️ Upload alone is NOT enough!                       │
│  ⚠️ Must complete full pipeline to be searchable      │
└─────────────────────────────────────────────────────┘
```

**Exam Pattern:** "New docs uploaded but answers still outdated" → Index not refreshed

### Retrieval Method Decision Tree

```javascript
What type of queries?
    │
    ├── Semantic/conceptual only → Dense (vector)
    │
    ├── Exact match/keywords only → Sparse (BM25)
    │
    └── Both types → Hybrid search

Results not relevant enough?
    │
    ├── Add re-ranker
    │
    └── Try query transformation (HyDE, multi-query)

Need access control?
    │
    └── Metadata filtering

Complex multi-hop questions?
    │
    └── Graph RAG or query decomposition
```

## References

- [Building Agentic RAG with Llamaindex - DeepLearning.AI](https://learn.deeplearning.ai/courses/building-agentic-rag-with-llamaindex)
- [Embeddings: A Deep Dive from Basics to Advanced Concepts](https://medium.com/@sharanharsoor/embeddings-a-deep-dive-from-basics-to-advanced-concepts-f092765476fc)

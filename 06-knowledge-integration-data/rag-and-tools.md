# RAG and Tools

## Embeddings

- Embeddings - a high-dimensional data as vectors in a lower-dimensional space. This matters because once words are vectors, we can mathematically measure how similar they are (using dot product or cosine similarity).
- For example, words like "cat" and "dog" can be represented as vectors that are close to each other in the vector space, reflecting their semantic similarity. This is in contrast to one-shot encoding, where each word is represented as a sparse binary vector, which fails to capture any semantic relationships.

## Embedding Models

- Embedding models are machine learning algorithms, often based on transformer architectures, that transform data like text, images, or audio into low-dimensional, dense vectors (numerical arrays).
- Embedding Models - Convert text → numbers (vectors) that capture meaning

| Model Type         | Examples                     | Dimensions | Use Case                        |
| ------------------ | ---------------------------- | ---------- | ------------------------------- |
| **Small/Fast**     | all-MiniLM-L6                | 384        | Quick prototypes, low resources |
| **Balanced**       | e5-base, bge-base            | 768        | Production balance              |
| **Large/Accurate** | e5-large, bge-large          | 1024       | High accuracy needs             |
| **Specialized**    | NeMo Retriever, Cohere embed | Varies     | Enterprise, multilingual        |

### Critical Rule

```
SAME embedding model for INDEXING and QUERYING!

Wrong:
  Index with: e5-large (1024 dims)
  Query with: MiniLM (384 dims)
  → Dimensions don't match, search fails!

Correct:
  Index with: e5-large
  Query with: e5-large (same model)
```

**Exam Pattern:** "Search returns irrelevant results after model update" → Embedding mismatch

## Router Engine

A **Router** is a component that decides **which tool, index, or model** to use based on the incoming query. Think of it as a traffic controller for your agent.

Without a router the agent would have to look through all documents.

```mermaid
graph TD
    Q["Query: 'revenue forecast'"]
    R["ROUTER<br/>Where should this go?"]
    IA["Index A<br/>(Finance)"]
    IB["Index B<br/>(Legal)"]
    TC["Tool C<br/>(Calculator)"]

    Q --> R
    R --> IA
    R --> IB
    R --> TC
```

### Router Types

| Router Type                           | How It Works                                                      | Use Case                          |
| ------------------------------------- | ----------------------------------------------------------------- | --------------------------------- |
| **LLM Selector Router (Most common)** | Uses an LLM to understand the query and pick the best destination | Complex queries needing reasoning |
| **Pydantic Router**                   | Structured output selection                                       | Type-safe routing                 |
| **Embedding Router**                  | Semantic similarity matching                                      | Fast, simple routing              |
| **Keyword/Rule-Based Router**         | Pattern matching on keywords                                      | Fastest, cheapest                 |

### LLM Selector Router (Most common)

Uses an LLM to understand the query and pick the best destination.

| Pros                        | Cons                   |
| --------------------------- | ---------------------- |
| Understands complex queries | Adds LLM latency       |
| Handles ambiguous intent    | Costs money (LLM call) |
| Can reason about routing    | Can make wrong choices |

### Embedding Router (Semantic Similarity)

Compares query embedding to description embeddings of each tool/index. Best for clear-cut categories, speed-critical applications.

```mermaid
graph TD
    Q["Query: 'revenue forecast'"]
    E["Embed query"]
    C["Compare to tool descriptions:<br/>- Financial data: 0.89 ✓<br/>- Legal contracts: 0.32<br/>- HR policies: 0.21"]
    R["Route to Finance<br/>(highest similarity)"]

    Q --> E
    E --> C
    C --> R
```

| Pros               | Cons                           |
| ------------------ | ------------------------------ |
| Fast (no LLM call) | Less nuanced understanding     |
| Cheap              | Depends on description quality |
| Deterministic      | Can't handle complex logic     |

### Pydantic Router (Structured Output)

| Pros                       | Cons                   |
| -------------------------- | ---------------------- |
| Type-safe, predictable     | Still needs LLM call   |
| Easy to debug              | Less flexible          |
| Includes confidence scores | Requires schema design |

**Best for:** Production systems needing reliability

### Keyword/Rule-Based Router

| Pros        | Cons                    |
| ----------- | ----------------------- |
| Fastest     | Misses semantic meaning |
| No cost     | Requires manual rules   |
| Predictable | Doesn't scale           |

### Exam Patterns for Routers

| Question Pattern                                        | Answer Points To    |
| ------------------------------------------------------- | ------------------- |
| "Agent needs to choose between multiple data sources"   | Router              |
| "Query should go to appropriate index based on content" | Router              |
| "Need fast routing without LLM"                         | Embedding Router    |
| "Complex routing decisions with reasoning"              | LLM Selector Router |
| "Type-safe, structured routing"                         | Pydantic Router     |

### Router vs Re-ranker

| Component     | When It Works    | What It Does            |
| ------------- | ---------------- | ----------------------- |
| **Router**    | BEFORE retrieval | Decides WHERE to search |
| **Re-ranker** | AFTER retrieval  | Reorders WHAT was found |

---

## similarity_top_k

`similarity_top_k` = Number of most similar chunks to retrieve

```
Query: "What is the company revenue?"
similarity_top_k=3 → Returns 3 most relevant chunks
similarity_top_k=10 → Returns 10 most relevant chunks
```

**Tradeoffs:**

| Low top_k (1-3)  | High top_k (10+)          |
| ---------------- | ------------------------- |
| Faster           | Slower                    |
| More focused     | More comprehensive        |
| May miss context | May include noise         |
| Lower cost       | Higher cost (more tokens) |

**Exam Tip:** If question mentions "missing relevant context" → increase top_k

---

## Standard RAG vs Agentic RAG

| Aspect      | Standard RAG                | Agentic RAG                                      |
| ----------- | --------------------------- | ------------------------------------------------ |
| Flow        | Query → Retrieve → Generate | Query → _Reason_ → _Tools_ → Retrieve → Generate |
| LLM Role    | Synthesis only              | **Reasoning + Tool selection**                   |
| Flexibility | Fixed pipeline              | Dynamic decisions                                |
| Tools       | Just retriever              | Multiple tools (search, calculate, API)          |
| Multi-step  | No                          | Yes (reasoning loop)                             |

**Exam Pattern:** "Agent needs to decide WHAT to retrieve and HOW" → Agentic RAG

---

## RAG Pipeline Components

```mermaid
graph LR
    subgraph indexing ["INDEXING"]
        D["Documents<br/>(Input)"]
        CH["Chunking"]
        EM["Embedding<br/>Model"]
        VS["Vector<br/>Store"]
    end

    subgraph querying ["QUERYING"]
        Q["Query"]
        R["Retrieval"]
        RR["Re-rank<br/>(optional)"]
        G["Generate<br/>(LLM)"]
    end

    D --> CH
    CH --> EM
    EM --> VS

    Q --> R
    R --> RR
    RR --> G
```

### Document Lifecycle in RAG

```mermaid
graph LR
    U["Upload"]
    P["Parse"]
    CH["Chunk"]
    E["Embed"]
    I["Index"]
    S["SEARCHABLE ✓"]

    U --> P
    P --> CH
    CH --> E
    E --> I
    I --> S
```

Upload alone is NOT enough! Must complete full pipeline to be searchable.

**Exam Pattern:** "New docs uploaded but answers still outdated" → Index not refreshed

---

## Chunking Strategies

| Strategy       | Description                                | Best For             |
| -------------- | ------------------------------------------ | -------------------- |
| **Fixed Size** | Split by character count (e.g., 512 chars) | Simple, fast         |
| **Sentence**   | Split at sentence boundaries               | Maintaining meaning  |
| **Paragraph**  | Split at paragraph breaks                  | Structured docs      |
| **Semantic**   | Split by meaning/topic                     | Best quality, slower |
| **Recursive**  | Try multiple splitters                     | General purpose      |

**Key Parameters:**

- `chunk_size`: Size of each chunk (tokens or chars)
- `chunk_overlap`: Overlap between chunks (prevents cutting context)

**Exam Pattern:** "Context is getting cut off" → Increase chunk_overlap

### Chunking Troubleshooting

| If you see...                 | The chunking problem is... | Fix                   |
| ----------------------------- | -------------------------- | --------------------- |
| "Context cut off"             | No overlap                 | Add `chunk_overlap`   |
| "Answers incomplete"          | Chunks too small           | Increase `chunk_size` |
| "Irrelevant content mixed in" | Chunks too big             | Decrease `chunk_size` |
| "Related info separated"      | Not semantic               | Use semantic chunking |

### Advanced Chunking Patterns

**1) Sentence Window Retrieval**

Concept: Index SMALL, Return BIG

```mermaid
graph TD
    DOC["Document: ... s1, s2, MATCH, s4, s5 ..."]
    IDX["Index: Small chunks<br/>for precise match"]
    FIND["Search finds MATCH<br/>(precise)"]
    RET["Return: s1, s2, MATCH, s4, s5<br/>(full context window)"]

    DOC --> IDX
    IDX --> FIND
    FIND --> RET
```

Benefit: Precise retrieval + full context!

**2) Parent-Child Retrieval**

```mermaid
graph TD
    P["PARENT CHUNK<br/>(Large - stored for context)"]
    C1["Child 1<br/>(indexed)"]
    C2["Child 2<br/>(indexed)"]
    C3["Child 3<br/>(indexed)"]

    P -->|contains| C1
    P -->|contains| C2
    P -->|contains| C3

    C2 -->|Search matches<br/>small, precise| M["Return Parent<br/>(large, contextual)"]
```

**Exam Pattern:** "Precise matching but need full context" → Parent-child or sentence window

**3) Hierarchical Chunking**

```mermaid
graph TD
    D["Document"]
    S1["Section 1"]
    P11["Paragraph 1.1"]
    P12["Paragraph 1.2"]
    S2["Section 2"]
    P21["Paragraph 2.1"]

    D --> S1
    D --> S2
    S1 --> P11
    S1 --> P12
    S2 --> P21
```

Maintains document structure for navigation.

---

## Metadata Filtering

Filter chunks **BEFORE** semantic search:

- Reduces search space
- More relevant results
- Faster queries
- Enables access control

**Exam Pattern:** "Agent should only access certain documents based on user role" → Metadata filtering

---

## Retrieval Methods

| Method                   | How It Works         | Pros                   | Cons                      |
| ------------------------ | -------------------- | ---------------------- | ------------------------- |
| **Dense (Vector)**       | Embedding similarity | Semantic understanding | Misses exact matches      |
| **Sparse (BM25/TF-IDF)** | Keyword matching     | Good for exact terms   | No semantic understanding |
| **Hybrid**               | Combines both        | Best of both worlds    | More complex              |

**Hybrid Search Formula:**

```
final_score = α × dense_score + (1-α) × sparse_score
```

**Exam Pattern:** "User searches for exact product code but also wants similar items" → Hybrid search

### Retrieval Method Decision Tree

```mermaid
graph TD
    Q["What type of queries?"]
    Q -->|Semantic/conceptual| Dense["Dense<br/>(vector)"]
    Q -->|Exact match/keywords| Sparse["Sparse<br/>(BM25)"]
    Q -->|Both types| Hybrid["Hybrid<br/>search"]

    R["Results not relevant enough?"]
    R -->|Add intelligence| Rerank["Add re-ranker"]
    R -->|Transform query| Transform["HyDE,<br/>multi-query"]

    AC["Need access control?"]
    AC -->|Yes| Filter["Metadata<br/>filtering"]

    MH["Complex multi-hop questions?"]
    MH -->|Yes| GraphRAG["Graph RAG or<br/>query decomposition"]

    Dense --> R
    Sparse --> R
    Hybrid --> R
    Rerank --> AC
    Transform --> AC
    Filter --> MH
```

---

## Re-ranking

Re-rank retrieved results for better relevance:

```mermaid
graph LR
    INIT["Initial retrieval<br/>(top_k=20)"]
    RR["Re-ranker"]
    FINAL["Final results<br/>(top_n=5)"]

    INIT --> RR
    RR --> FINAL
```

**Re-ranking Models:**

- Cross-encoders (more accurate, slower)
- Cohere Rerank
- NVIDIA NeMo Retriever Re-ranker

**When to use:**

- Initial retrieval returns many results
- Need higher precision
- Quality > Speed

---

## Vector Databases & Indexing Algorithms

### Popular Vector Stores

| Vector DB    | Type     | Best For                                 |
| ------------ | -------- | ---------------------------------------- |
| **FAISS**    | Library  | Local, research, fast                    |
| **Milvus**   | Database | Production, scalable, NVIDIA partnership |
| **Pinecone** | Managed  | Easy setup, serverless                   |
| **Weaviate** | Database | Hybrid search built-in                   |
| **Chroma**   | Library  | Lightweight, prototypes                  |
| **Qdrant**   | Database | Filtering, payloads                      |

### Indexing Algorithms

| Algorithm              | How It Works            | Accuracy | Speed     | Use Case           |
| ---------------------- | ----------------------- | -------- | --------- | ------------------ |
| **Flat (Brute Force)** | Compare to EVERY vector | 100%     | Very slow | <10K vectors       |
| **HNSW**               | Graph-based approximate | ~95-99%  | Very fast | Most production    |
| **IVF**                | Cluster-based search    | ~90-95%  | Fast      | Billion-scale      |
| **PQ**                 | Compressed vectors      | Lower    | Fast      | Memory-constrained |

**Exam Pattern:** "Fast search on millions of vectors" → HNSW or IVF

---

## Query Transformation Techniques

### HyDE (Hypothetical Document Embeddings)

```mermaid
graph TD
    PROB["Problem: Query is short<br/>documents are long"]
    Q["Query: 'What is RAG?'"]
    LLM["LLM generates<br/>hypothetical answer"]
    HYPO["'RAG is a technique that<br/>combines retrieval....'"]
    EMB["Embed THIS hypothetical doc"]
    SEARCH["Search now matches<br/>document style!"]

    PROB --> Q
    Q --> LLM
    LLM --> HYPO
    HYPO --> EMB
    EMB --> SEARCH
```

**When to use:** Short queries, document-style answers expected

### Multi-Query / Query Expansion

```mermaid
graph TD
    ORIG["Original: 'RAG best practices'"]
    GEN["Generate variations:<br/>- How to implement RAG effectively<br/>- RAG optimization techniques<br/>- Common RAG patterns"]
    SEARCH["Search with ALL queries"]
    COMBINE["Combine results"]

    ORIG --> GEN
    GEN --> SEARCH
    SEARCH --> COMBINE
```

**When to use:** Ambiguous queries, comprehensive search needed

### Query Decomposition

```mermaid
graph TD
    COMPLEX["Complex: 'Compare RAG vs<br/>fine-tuning for customer service'"]
    DECOMP["Decompose:<br/>- Advantages of RAG?<br/>- Advantages of fine-tuning?<br/>- Chatbot requirements?"]
    SEARCH["Search each subquery"]
    COMBINE["Combine answers"]

    COMPLEX --> DECOMP
    DECOMP --> SEARCH
    SEARCH --> COMBINE
```

**Exam Pattern:** "Complex multi-part questions failing" → Query decomposition

---

## Adaptive RAG

**Adaptive RAG** dynamically selects different RAG strategies based on query complexity or type. Think of it as a "smart router" that routes to entirely **different processing pipelines**.

```mermaid
graph TD
    Q["User Query"]
    CLASSIFY["Complexity Classifier<br/>(LLM or rule-based)"]

    SIMPLE["SIMPLE<br/>Direct LLM call<br/>(no RAG)"]
    MODERATE["MODERATE<br/>Standard RAG<br/>pipeline"]
    COMPLEX["COMPLEX<br/>Multi-hop RAG +<br/>reasoning"]

    Q --> CLASSIFY
    CLASSIFY -->|Simple| SIMPLE
    CLASSIFY -->|Moderate| MODERATE
    CLASSIFY -->|Complex| COMPLEX
```

| Pattern          | What It Does                                                                   | Key Trigger                                                             |
| ---------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| **Multi-Query**  | Generates multiple variations of SAME query, retrieves for ALL, unions results | "Rephrase", "multiple perspectives", "same question different ways"     |
| **Adaptive RAG** | Routes to DIFFERENT strategies based on complexity                             | "Simple vs complex", "different approaches", "complexity-based routing" |

**Memory trick:**

- Multi-Query = **M**ultiple versions of same question
- Adaptive RAG = **A**djusts strategy based on complexity

---

## RAG Fusion (Reciprocal Rank Fusion)

The whole point of RAG Fusion is getting **richer, more diverse retrieval** from multiple queries. But richer retrieval means messier raw results. You NEED intelligence to make sense of it - that's what the LLM does.

**RRF Formula:**

```
RRF_score = Σ (1 / (k + rank_i))
where k = constant (usually 60)
```

**Why it works:** Documents ranked highly by multiple queries get boosted

---

## Self-RAG & CRAG Patterns

### Self-RAG (Self-Reflective RAG)

Agent critiques its own retrieval and generation:

```mermaid
graph TD
    Q["Query"]
    R["Retrieve"]
    G["Generate"]
    CRIT["Self-Critique:<br/>Is this grounded?<br/>Is retrieval needed?"]
    REFINE["Refine if needed"]
    ANSWER["Final Answer"]

    Q --> R
    R --> G
    G --> CRIT
    CRIT --> REFINE
    REFINE --> ANSWER
```

**Key self-assessment tokens:**

- `[Retrieve]` - Do I need to retrieve?
- `[IsRel]` - Is retrieved doc relevant?
- `[IsSup]` - Is response supported by docs?
- `[IsUse]` - Is response useful?

### CRAG (Corrective RAG)

CRAG has **web fallback** for bad retrieval:

```mermaid
graph TD
    Q["Query"]
    R["Retrieve"]
    EVAL["Evaluate Relevance"]

    CORRECT["Correct"]
    AMBIG["Ambiguous"]
    INCORRECT["Incorrect"]

    USE["Use docs"]
    REFINE["Refine search"]
    WEB["Web search<br/>(fallback)"]

    Q --> R
    R --> EVAL
    EVAL -->|Correct| CORRECT
    EVAL -->|Ambiguous| AMBIG
    EVAL -->|Incorrect| INCORRECT

    CORRECT --> USE
    AMBIG --> REFINE
    INCORRECT --> WEB
```

**Exam Pattern:** "Agent should verify retrieval quality" → Self-RAG or CRAG

---

## The "Lost in the Middle" Problem

**Problem:** LLMs attend more to START and END, less to MIDDLE of context

```mermaid
graph LR
    D1["[Doc1]"]
    D2["[Doc2]"]
    D3["[Doc3]"]
    D4["[Doc4]"]
    D5["[Doc5]"]
    D6["[Doc6]"]

    D1 -.->|Attended| ATT1["✓ HIGH"]
    D3 -.->|Less attended| ATT3["⚠ LOW"]
    D6 -.->|Attended| ATT6["✓ HIGH"]
```

**Solutions:**

| Solution             | How                                |
| -------------------- | ---------------------------------- |
| **Reorder results**  | Put most relevant at start AND end |
| **Reduce context**   | Only include top 3-5, not 10+      |
| **Summarize middle** | Compress middle docs               |
| **Use re-ranker**    | Ensure best docs are first         |

**Exam Pattern:** "Important info in middle being ignored" → Lost in middle problem

---

## Agent Reasoning Loop (ReAct)

```mermaid
graph TD
    Q["Query"]
    T1["Thought"]
    A["Action"]
    O["Observe"]
    T2["Thought"]
    DONE["Done?"]
    ANS["Answer"]

    Q --> T1
    T1 --> A
    A --> O
    O --> DONE
    DONE -->|No| T2
    T2 --> A
    DONE -->|Yes| ANS
```

**FunctionCallingAgentWorker:**

- Implements the ReAct loop
- Manages tool selection
- Handles multi-step reasoning
- Maintains conversation state

---

## RAG Evaluation Metrics

### Retrieval Metrics

| Metric          | What It Measures                | Interpretation                |
| --------------- | ------------------------------- | ----------------------------- |
| **Precision@K** | Relevant in top K / K           | How many retrieved are useful |
| **Recall@K**    | Relevant found / Total relevant | Did we find all useful docs   |
| **MRR**         | 1/rank of first relevant        | How quickly we find relevant  |
| **NDCG**        | Normalized ranking quality      | Are better docs ranked higher |

### Generation Metrics

| Metric                | What It Measures                      |
| --------------------- | ------------------------------------- |
| **Faithfulness**      | Is answer grounded in retrieved docs? |
| **Answer Relevance**  | Does answer address the question?     |
| **Context Relevance** | Are retrieved docs relevant to query? |

### RAGAS Framework

```mermaid
graph TD
    RAG["RAGAS Framework"]
    F["Faithfulness<br/>Answer supported by context?"]
    AR["Answer Relevance<br/>Addresses question?"]
    CP["Context Precision<br/>Docs relevant to query?"]
    CR["Context Recall<br/>All needed info retrieved?"]

    RAG --> F
    RAG --> AR
    RAG --> CP
    RAG --> CR
```

**Exam Pattern:** "Evaluate RAG system quality" → Faithfulness + Context Relevance + Answer Relevance

---

## Multi-Modal RAG

### Handling Different Content Types

| Content Type | Approach                             |
| ------------ | ------------------------------------ |
| **Text**     | Standard text embeddings             |
| **Images**   | CLIP or vision model embeddings      |
| **Tables**   | Serialize to text OR keep structured |
| **PDFs**     | OCR + layout analysis → chunks       |
| **Code**     | Code-specific embeddings             |

### Table Handling Strategies

| Strategy       | How                         | Best For              |
| -------------- | --------------------------- | --------------------- |
| **Serialize**  | Convert table to text       | Simple tables         |
| **Structured** | Keep as JSON/rows           | Query-able tables     |
| **Summary**    | LLM describes table         | Context understanding |
| **Hybrid**     | Store both text + structure | Flexible queries      |

---

## Graph RAG (Knowledge Graphs + RAG)

```mermaid
graph TD
    subgraph Traditional["Traditional RAG"]
        Q1["Query"]
        VS["Vector search"]
        CH["Chunks"]
        A1["Answer"]
        Q1 --> VS
        VS --> CH
        CH --> A1
    end

    subgraph GraphBased["Graph RAG"]
        Q2["Query"]
        EE["Extract entities"]
        GT["Graph traversal"]
        RE["Related entities"]
        VS2["Vector search<br/>on subgraph"]
        A2["Richer,<br/>connected context"]
        Q2 --> EE
        EE --> GT
        GT --> RE
        RE --> VS2
        VS2 --> A2
    end
```

**When to use Graph RAG:**

- Multi-hop reasoning ("Who is the CEO's boss?")
- Relationship queries ("What products does Company X make?")
- Interconnected knowledge

**NVIDIA component:** **RAPIDS cuGraph** for graph processing

---

## RAG vs Fine-tuning

| Aspect               | RAG                     | Fine-tuning            |
| -------------------- | ----------------------- | ---------------------- |
| **Knowledge update** | Easy (update docs)      | Hard (retrain)         |
| **Cost**             | Lower                   | Higher                 |
| **Hallucination**    | Reduced (grounded)      | Can still hallucinate  |
| **Latency**          | Higher (retrieval step) | Lower                  |
| **Best for**         | Dynamic knowledge       | Style/behavior changes |

**Exam Pattern:** "Knowledge changes frequently" → RAG
**Exam Pattern:** "Need consistent response style" → Fine-tuning

---

## NVIDIA-Specific RAG Components

| NVIDIA Component   | RAG Role                                                         |
| ------------------ | ---------------------------------------------------------------- |
| **AIQ Toolkit**    | Enterprise connectors (Confluence, SharePoint), memory providers |
| **NeMo Retriever** | Embedding models, re-rankers                                     |
| **NIM**            | Deploy embedding models at scale                                 |
| **Triton**         | Serve retrieval models efficiently                               |
| **RAPIDS cuGraph** | Knowledge graph integration                                      |

**Exam Pattern:** "Enterprise RAG with existing knowledge bases" → AIQ Toolkit connectors

---

## Production RAG Considerations

### Caching Strategies

| Cache Type          | How It Works                              |
| ------------------- | ----------------------------------------- |
| **Query Cache**     | Same exact query → cached results         |
| **Semantic Cache**  | Similar queries → cached results          |
| **Embedding Cache** | Skip embedding computation for known text |

### Latency Optimization

| Technique                     | Impact                     |
| ----------------------------- | -------------------------- |
| **Embedding caching**         | Skip embedding computation |
| **Result caching**            | Skip retrieval entirely    |
| **Async retrieval**           | Parallel processing        |
| **Smaller top_k**             | Less to process            |
| **Approximate search (HNSW)** | vs brute force             |
| **Metadata filtering**        | Smaller search space       |

### Cost Management

```mermaid
graph TD
    COSTS["Costs in RAG"]
    C1["Embedding API calls<br/>(per token)"]
    C2["Vector DB storage<br/>(per vector)"]
    C3["LLM API calls<br/>(per token in context)"]

    REDUCE["Reduce costs by:"]
    R1["Cache embeddings"]
    R2["Reduce chunk overlap<br/>(carefully)"]
    R3["Lower top_k"]
    R4["Compress context<br/>before LLM"]

    COSTS --> C1
    COSTS --> C2
    COSTS --> C3

    REDUCE --> R1
    REDUCE --> R2
    REDUCE --> R3
    REDUCE --> R4
```

---

## RAG Anti-Patterns (What NOT to Do)

| Anti-Pattern                   | Problem                    | Fix                          |
| ------------------------------ | -------------------------- | ---------------------------- |
| **Different embedding models** | Dimension mismatch         | Same model for index & query |
| **No chunk overlap**           | Lost context at boundaries | Add 10-20% overlap           |
| **Too many results to LLM**    | Context overflow, noise    | Re-rank, lower top_k         |
| **No re-indexing schedule**    | Stale answers              | Automated re-indexing        |
| **Ignoring metadata**          | Can't filter or scope      | Add metadata during indexing |
| **Flat file storage**          | No relationships           | Use proper vector DB         |

---

## Common RAG Failures & Fixes

| Problem                | Symptom                    | Fix                       |
| ---------------------- | -------------------------- | ------------------------- |
| **Bad chunking**       | Context cut off            | Adjust chunk_size/overlap |
| **Low top_k**          | Missing relevant info      | Increase similarity_top_k |
| **No re-ranking**      | Irrelevant results at top  | Add re-ranker             |
| **Embedding mismatch** | Semantically wrong results | Better embedding model    |
| **Stale index**        | Outdated answers           | Re-index regularly        |
| **No metadata**        | Can't filter results       | Add metadata to chunks    |

---

## Exam Question Patterns for RAG

| Question Type        | Look For        | Answer Points To            |
| -------------------- | --------------- | --------------------------- |
| "Missing context"    | Retrieval issue | top_k, chunking             |
| "Slow retrieval"     | Performance     | Caching, index optimization |
| "Wrong documents"    | Relevance       | Re-ranking, hybrid search   |
| "Access control"     | Security        | Metadata filtering          |
| "Outdated answers"   | Freshness       | Re-indexing, sync schedules |
| "Enterprise sources" | Integration     | AIQ Toolkit connectors      |

---

## References

- [Building Agentic RAG with LlamaIndex](https://learn.deeplearning.ai/courses/building-agentic-rag-with-llamaindex)

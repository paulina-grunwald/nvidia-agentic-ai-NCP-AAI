# RAG - Retrieval Argumented Generation

### Embedings

- **Embedding** is a high-dimensional data as vectors in a lower-dimensional space => This matters because once words are vectors, we can mathematically measure how similar they are (using dot product or cosine similarity).
- For example, words like "cat" and "dog" can be represented as vectors that are close to each other in the vector space, reflecting their semantic similarity. This is in contrast to one-hot encoding, where each word is represented as a sparse binary vector, which fails to capture any semantic relationships.
- __One-hot encoding__:
  - one-hot encoding is just an ID system (like assigning each word a number), while dense embeddings actually encode what words mean relative to each other.
  ```
  king  = [1, 0, 0, 0, 0]   ← the 1 is at position 0
  queen = [0, 1, 0, 0, 0]   ← the 1 is at position 1
  (1×0) + (0×1) + (0×0) + (0×0) + (0×0) = 0
  ```
- __Dense embeddings__
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

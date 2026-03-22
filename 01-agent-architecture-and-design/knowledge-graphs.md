# Knowledge Graphs for Relational Reasoning (Objective 1.7)

## What Are Knowledge Graphs?

A knowledge graph is a structured representation of real-world entities and the relationships between them. Think of it as a web of facts: "Company X → employs → Person Y" or "Drug A → treats → Disease B".

Unlike vector databases that store chunks of text as embeddings (good for semantic similarity search), knowledge graphs store explicit relationships. This makes them powerful for answering questions that require reasoning across connections.

```
Example:
  [Tesla] --manufactures--> [Model 3]
  [Tesla] --CEO--> [Elon Musk]
  [Model 3] --category--> [Electric Vehicle]
  [Electric Vehicle] --requires--> [Battery]
  [Battery] --supplied_by--> [Panasonic]
```

With this graph, an agent can reason: "Who supplies batteries for Tesla's Model 3?" by traversing: Model 3 → Electric Vehicle → Battery → Panasonic. A vector search would struggle with this kind of multi-hop reasoning.

---

## Why Agents Need Knowledge Graphs

| Capability | Vector DB alone | With Knowledge Graph |
|---|---|---|
| Semantic similarity search | ✅ Great | ✅ Also supported |
| Multi-hop reasoning | ❌ Poor | ✅ Strong |
| Explicit relationships | ❌ Implicit only | ✅ Explicit edges |
| Structured queries | ❌ Limited | ✅ Graph queries (Cypher, SPARQL) |
| Explainability | ❌ Black box embeddings | ✅ Traceable reasoning paths |
| Entity disambiguation | ❌ Hard | ✅ Unique entity nodes |

Key insight: knowledge graphs complement vector DBs. They don't replace them. Many production systems use both (see Graph RAG below).

---

## Core Concepts

### Triple Structure

Every fact in a knowledge graph is stored as a triple:

```
(Subject) --[Predicate]--> (Object)

Examples:
(Paris)    --[capital_of]-->  (France)
(Python)   --[created_by]--> (Guido van Rossum)
(NeMo)     --[developed_by]--> (NVIDIA)
(Triton)   --[optimizes]-->   (Inference)
```

### Nodes and Edges

- **Nodes** (entities): People, places, products, concepts, documents
- **Edges** (relationships): How entities relate to each other, with labels and sometimes properties
- **Properties**: Attributes on nodes or edges (e.g., a "person" node might have name, age, role)

### Schema vs Schema-less

- **Schema-enforced** (ontology): Pre-defined entity types and allowed relationships. Stricter but more consistent. Good for enterprise use.
- **Schema-less**: Any entity can connect to any other. More flexible but can get messy. Good for rapid prototyping.

---

## Graph Databases

| Database | Query Language | Strengths | Use Case |
|---|---|---|---|
| **Neo4j** | Cypher | Most popular, rich ecosystem, good visualization | General purpose, enterprise |
| **Amazon Neptune** | Gremlin, SPARQL | Managed AWS service, scales well | Cloud-native applications |
| **ArangoDB** | AQL | Multi-model (graph + document + key-value) | Flexible data needs |
| **Dgraph** | GraphQL+- | Distributed, horizontal scaling | Large-scale graphs |
| **NVIDIA cuGraph** | Python API | GPU-accelerated graph analytics | High-performance graph computation |

### Cypher Query Example (Neo4j)

```cypher
// Find all products manufactured by Tesla
MATCH (c:Company {name: "Tesla"})-[:MANUFACTURES]->(p:Product)
RETURN p.name, p.year

// Multi-hop: Find suppliers of components used in Tesla products
MATCH (c:Company {name: "Tesla"})-[:MANUFACTURES]->(p:Product)
      -[:USES]->(comp:Component)-[:SUPPLIED_BY]->(s:Supplier)
RETURN p.name, comp.name, s.name
```

An agent can generate these Cypher queries dynamically as tool calls, similar to how it generates SQL for relational databases.

---

## Entity-Relationship Modeling for Agent Knowledge

When building a knowledge graph for an agent system, think about what entities and relationships the agent needs to reason about.

### Example: Customer Service Agent

```
Entities:
  - Customer (name, account_id, tier)
  - Product (name, SKU, category)
  - Order (order_id, date, status)
  - Issue (type, severity, resolution)

Relationships:
  - Customer --PLACED--> Order
  - Order --CONTAINS--> Product
  - Customer --REPORTED--> Issue
  - Issue --RELATED_TO--> Product
  - Issue --RESOLVED_BY--> Resolution
```

With this graph, the agent can answer: "Has this customer had issues with this product before?" by traversing Customer → Issue → Product and checking if the product matches.

### Example: Financial Fraud Detection Agent

This maps to the NVIDIA suggested reading "Catch Me If You Can: A Multi-Agent Framework for Financial Fraud Detection":

```
Entities:
  - Account, Transaction, Person, Merchant, Device

Relationships:
  - Person --OWNS--> Account
  - Account --SENT_TO--> Account (via Transaction)
  - Transaction --OCCURRED_AT--> Merchant
  - Person --USES--> Device
```

Fraud detection relies on finding unusual patterns in the graph: circular money flows, shared devices across unrelated accounts, sudden new connections.

---

## Knowledge Graph Construction from Unstructured Data

Agents often need to build knowledge graphs from text documents, not structured databases. This is a key enterprise use case.

### Pipeline

```
Raw Text → NER (Named Entity Recognition) → Entity Extraction
                                                    ↓
                                          Relation Extraction
                                                    ↓
                                          Entity Resolution (deduplication)
                                                    ↓
                                          Knowledge Graph
```

### Steps

1. **Entity Extraction (NER)**: Use an LLM or NER model to identify entities in text
   - "NVIDIA released TensorRT-LLM for optimized inference" → [NVIDIA: Company], [TensorRT-LLM: Product], [inference: Concept]

2. **Relation Extraction**: Determine how entities are related
   - "NVIDIA released TensorRT-LLM" → (NVIDIA) --released--> (TensorRT-LLM)

3. **Entity Resolution**: Merge duplicates
   - "NVIDIA Corp", "NVIDIA", "Nvidia Corporation" → single node

4. **Graph Population**: Insert triples into the graph database

LLMs are increasingly used for all three steps. You can prompt an LLM: "Extract all entities and relationships from this text as triples" and get structured output.

---

## Graph RAG (Knowledge Graph + RAG)

Graph RAG combines the strengths of both approaches: vector search for semantic retrieval and graph traversal for relational reasoning.

### How It Works

```
User Question
     ↓
┌────────────┐    ┌────────────────┐
│ Vector      │    │ Graph           │
│ Search      │    │ Traversal       │
│ (semantic)  │    │ (relational)    │
└──────┬─────┘    └───────┬────────┘
       │                  │
       └────────┬─────────┘
                ↓
        Merged Context
                ↓
        LLM Generates Answer
```

### Example

Question: "What are the side effects of Drug X for patients with Condition Y?"

- **Vector search**: Retrieves relevant medical documents mentioning Drug X
- **Graph traversal**: Follows (Drug X) → TREATS → (Condition Y) → SIDE_EFFECTS → [list], and also checks (Drug X) → INTERACTS_WITH → [other drugs the patient takes]
- **Merged context**: Both document chunks and graph relationships are sent to the LLM
- **Result**: More complete, accurate answer with explicit reasoning chains

### Why Graph RAG Matters for the Exam

The exam blueprint specifically mentions "integrate knowledge graphs to enable relational reasoning." Graph RAG is the production pattern that achieves this. It addresses limitations of pure vector-based RAG:

- **Multi-hop questions**: "Who manages the team that built the feature causing the outage?" requires traversing relationships
- **Structured constraints**: "Find all products in category X with recall history" needs explicit filtering
- **Explainability**: The graph path provides an auditable reasoning chain

---

## Relational Reasoning with Knowledge Graphs

Relational reasoning means drawing conclusions by following and combining relationships. This is what distinguishes knowledge graphs from simpler retrieval methods.

### Types of Relational Reasoning

| Type | Example | Graph Pattern |
|---|---|---|
| **Direct lookup** | "Who is the CEO of NVIDIA?" | Single hop: (NVIDIA) --CEO--> (?) |
| **Multi-hop** | "What university did NVIDIA's CEO attend?" | Two hops: (NVIDIA) --CEO--> (Person) --ATTENDED--> (?) |
| **Aggregation** | "How many products does NVIDIA make?" | Count: (NVIDIA) --MAKES--> count(*) |
| **Path finding** | "How are Company A and Company B connected?" | Shortest path between two nodes |
| **Pattern matching** | "Find all circular transaction chains" | Cycle detection (fraud) |
| **Inference** | "If A supplies B and B supplies C, then A indirectly supplies C" | Transitive closure |

### Agent Tool Integration

An agent can use a knowledge graph as a tool, just like it uses search or calculator:

```
Thought: I need to find how NVIDIA's products relate to autonomous vehicles.
Action: query_knowledge_graph(
    query="MATCH (n:Company {name:'NVIDIA'})-[:MAKES]->(p:Product)
           -[:USED_IN]->(d:Domain {name:'Autonomous Vehicles'})
           RETURN p.name"
)
Observation: ["DRIVE Orin", "DRIVE Thor", "DriveWorks SDK"]
Thought: I now have the list of NVIDIA products used in autonomous vehicles.
```

---

## Integration with LLM Agents

### Pattern 1: Graph as Tool

The agent calls the graph database as an external tool (same as calling an API):

```python
tools = [
    Tool(name="graph_query", func=neo4j_client.run,
         description="Query the knowledge graph using Cypher")
]
```

### Pattern 2: Graph-Enhanced Context

Before each LLM call, relevant graph context is retrieved and injected into the prompt:

```python
# Get entity context from graph
entity_context = graph.get_neighbors("NVIDIA", depth=2)
# Add to LLM prompt
prompt = f"Context from knowledge graph:\n{entity_context}\n\nUser question: {query}"
```

### Pattern 3: Graph-Guided RAG

Use the graph to determine WHICH documents to retrieve, then use vector search within those:

```python
# Step 1: Graph query to find relevant entities
related_entities = graph.query("MATCH (n)-[*1..3]-(target) WHERE n.name=$query RETURN target")
# Step 2: Use entity names to filter vector search
docs = vector_store.search(query, filter={"entities": related_entities})
```

---

## NVIDIA cuGraph

NVIDIA provides GPU-accelerated graph analytics through cuGraph (part of RAPIDS). While not a full graph database, it's relevant for:

- Large-scale graph analytics at GPU speed
- PageRank, community detection, shortest paths on massive graphs
- Integration with NVIDIA's AI pipeline (cuDF → cuGraph → ML models)
- Processing billion-edge graphs that would be too slow on CPU

---

## Exam-Style Questions

**Q1: A financial services agent needs to detect money laundering by finding circular transaction chains. Which technology is most appropriate?**
- A) Vector database with semantic search
- B) Knowledge graph with cycle detection
- C) SQL database with JOIN queries
- D) Document store with full-text search

**Answer: B** -Cycle detection in a knowledge graph is the standard approach for finding circular transaction patterns. Vector search finds semantically similar documents, not structural patterns.

**Q2: What is the main advantage of Graph RAG over traditional vector-based RAG?**
- A) Lower latency
- B) Better multi-hop reasoning through explicit relationships
- C) Smaller storage requirements
- D) Simpler implementation

**Answer: B** -Graph RAG enables multi-hop reasoning by combining semantic search with explicit relationship traversal. Traditional RAG struggles with questions requiring multiple reasoning steps.

**Q3: An agent needs to answer "What university did NVIDIA's CEO attend?" from a knowledge graph. How many hops are required?**
- A) 1 hop
- B) 2 hops
- C) 3 hops
- D) Cannot be determined

**Answer: B** -Two hops: (NVIDIA) → CEO → (Person) → ATTENDED → (University)

**Q4: Which approach best combines vector search and knowledge graph capabilities?**
- A) Replace vector DB with graph DB
- B) Use Graph RAG to merge semantic and relational retrieval
- C) Store embeddings inside the graph database
- D) Use only graph queries for all retrieval

**Answer: B** -Graph RAG merges both approaches, using vector search for semantic relevance and graph traversal for relational reasoning.

---

## Key Takeaways for NVIDIA Certification

1. Knowledge graphs store entities and relationships as triples (subject → predicate → object)
2. They enable multi-hop relational reasoning that vector databases alone cannot do
3. Graph RAG combines vector search + graph traversal for the best of both worlds
4. Agents use knowledge graphs as tools (generating Cypher/SPARQL queries) or as context enhancers
5. Construction from unstructured data follows: NER → Relation Extraction → Entity Resolution → Graph
6. NVIDIA cuGraph provides GPU-accelerated graph analytics for large-scale processing
7. Common exam scenarios: fraud detection (cycle finding), supply chain (path tracing), customer service (entity lookup)

# NVIDIA Component Decision Guide

## Your Blueprint for Choosing the Right Tool

This is the quickest way to nail component selection questions on the exam. Bookmark this page.

---

## The One Table That Rules Them All

Keep this table front-and-center. Nine out of ten exam questions boil down to picking the right component:

| Scenario                                                   | Component                    |
| ---------------------------------------------------------- | ---------------------------- |
| Cleaning raw training data (dedup, filtering, PII removal) | NeMo Curator                 |
| Fine-tuning a model for your domain                        | NeMo Framework               |
| Making inference faster (lower latency, higher throughput) | TensorRT-LLM                 |
| Deploying a single model as a containerized API            | NIM                          |
| Serving multiple models with auto-batching and concurrency | Triton Inference Server      |
| Building agents that work with multiple frameworks         | NeMo Agent Toolkit (NAT)     |
| Adding content safety without modifying the model          | NeMo Guardrails + Colang     |
| Enterprise RAG with long-term memory and connectors        | AIQ Toolkit                  |
| Monitoring GPU health in production                        | DCGM + Grafana               |
| Finding optimal batch size and concurrency settings        | Triton Model Analyzer        |
| Deep GPU debugging (kernel profiling, timeline)            | Nsight Systems               |
| GPU-accelerated data science (SQL, ML, graphs)             | RAPIDS (cuDF, cuML, cuGraph) |
| Building embeddings and re-ranking for RAG                 | NeMo Retriever               |
| Aligning models with human feedback (RLHF)                 | NeMo Aligner                 |

---

## Keyword Triggers

When you see these words in a question, your answer is almost always locked in:

**Speed, Latency, Throughput, Faster Inference** → TensorRT-LLM

- The exam code for "we need inference faster"

**Scalability, Concurrent Requests, Multiple Models, Batch Processing** → Triton Inference Server

- Triton is the serving platform. Remember: it's infrastructure, not a model

**Insufficient, Lacking, Poor Understanding, Domain-Specific Knowledge** → NeMo Framework (fine-tuning)

- Model capability gap = training/fine-tuning

**Data Quality, Deduplication, PII, Filtering, Raw Datasets** → NeMo Curator

- The data prep tool. Always appears before training

**Works with LangChain AND CrewAI, Framework-agnostic, Unified Orchestration** → NAT or AIQ Toolkit

- NVIDIA's answer to "our agents are stuck using different frameworks"

**Topic Control, Content Policy, Block Discussions, Safety Without Model Changes** → NeMo Guardrails

- Guardrails are your gating layer before inference

**Model Swap Without Code Changes, Runtime Flexibility** → NIM

- NIM is the container. Drop it in, switch models, done

**Cross-Session Memory, Enterprise Connectors, Knowledge Base** → AIQ Toolkit

- AIQ has built-in memory providers and data connectors

**GPU Metrics, Production Alerts, System Health** → DCGM

- For monitoring. Not for tuning, not for optimization

---

## The Traps

These are the wrong answers hiding in plain sight:

1. **Triton does NOT...**
   - Handle memory or conversation state (that's AIQ)
   - Train or fine-tune models (that's NeMo)
   - Optimize kernel-level performance (that's TensorRT-LLM)
   - Manage agent orchestration (that's NAT)

2. **NIM does NOT...**
   - Train anything
   - Orchestrate agents
   - Store long-term memory
   - Handle safety (that's Guardrails)

3. **NeMo Framework is NOT...**
   - A production serving platform (that's Triton)
   - An optimization tool (that's TensorRT)
   - A containerized deployment (that's NIM)

4. **TensorRT-LLM does NOT...**
   - Serve models to multiple clients (that's Triton)
   - Handle batching and concurrency automatically (Triton does)
   - Deploy models without Triton (you optimize with TensorRT, then serve with Triton)

5. **Dynamic Batching** is a TRITON feature, not NeMo

6. **NeMo CREATES, NIM DEPLOYS**
   - Train or fine-tune in NeMo → containerize as NIM → serve on Triton

---

## The NVIDIA-Native Stack Pattern

The exam strongly prefers answers that use NVIDIA's full stack:

```
Agent Request
    ↓
NeMo Agent Toolkit (orchestration, memory, evaluation)
    ↓
AIQ Toolkit (enterprise connectors, long-term memory)
    ↓
NeMo Guardrails (safety gates)
    ↓
NIM (containerized inference)
    ↓
Triton Inference Server (serving at scale)
    ↓
DCGM (production monitoring)
```

When a question asks "how do we build this enterprise agent system," the right answer chains these together. Non-NVIDIA answers (LangChain, Redis, hand-rolled Python) are almost always wrong when an NVIDIA alternative exists on the exam.

---

## Capability vs Infrastructure Decision

This is how you differentiate between training and deployment questions:

**The model doesn't understand your domain?** → Capability gap → NeMo Framework (fine-tuning)

- Fix the model itself

**The model is slow or hard to deploy?** → Infrastructure gap → TensorRT-LLM + Triton

- Optimize and serve the model

**The model lacks specific knowledge?** → Knowledge gap → RAG with NeMo Retriever + AIQ Toolkit

- Augment the model with context

---

## Profiling and Monitoring (Quick Hit)

When questions ask about troubleshooting:

| Question Mentions                                       | Answer                |
| ------------------------------------------------------- | --------------------- |
| Debug GPU perf, kernel profiling, timeline, what's slow | Nsight Systems        |
| Monitor production, GPU metrics, utilization, alerts    | DCGM + Grafana        |
| Optimize batch size, best concurrency, serving config   | Triton Model Analyzer |
| Data science, SQL on GPU, ML algorithms, graphs         | RAPIDS                |

---

## Exam Questions

The exam tests one of three decision types:

1. **Which tool for this job?** (40% of questions)
   - "We need to clean our training data."
   - Look at the scenario table. Answer: NeMo Curator.

2. **Why not use X?** (35% of questions)
   - "Why can't we use NIM to handle agent orchestration?"
   - Remember: NIM is for inference containerization only. It's not an orchestration layer.

3. **What's missing from our stack?** (25% of questions)
   - "We're using Triton and NeMo. We need production monitoring. What's the gap?"
   - Answer: DCGM for GPU monitoring.

---

## Practice MCQs

### Question 1

Your team has raw customer feedback data with 40% duplicates and embedded PII. Before training a domain-specific model, what's the first step?

A) Import directly into NeMo Framework and set deduplication flags
B) Use NeMo Curator to clean, deduplicate, and filter PII
C) Implement a custom Python script with pandas to handle duplicates
D) Use AIQ Toolkit's data connectors to load and filter the data

**Answer: B**

- Curator is built exactly for pre-training data prep
- A is wrong: NeMo Framework doesn't have dedup; that's Curator's job
- C is wrong: NVIDIA tool exists, so custom code isn't the exam answer
- D is wrong: AIQ is for enterprise connectors and agent memory, not data cleaning

---

### Question 2

Your fine-tuned LLM is accurate but takes 500ms per inference call. Users need <100ms latency. What's your next step?

A) Re-train the model with TensorRT-LLM as the framework
B) Optimize the model with TensorRT-LLM, then serve on Triton Inference Server
C) Switch to a smaller model and use NIM for deployment
D) Increase your GPU memory and use Triton dynamic batching

**Answer: B**

- TensorRT-LLM optimizes; Triton serves—both are needed
- A is wrong: TensorRT is not a training framework
- C is wrong: Changing model size doesn't address latency with your specific model
- D is wrong: GPU memory and batching won't get you from 500ms to 100ms

---

### Question 3

You're building an enterprise RAG system where agents need to access company databases, maintain conversation history, and retrieve relevant documents. The agent framework must support both LangChain and CrewAI. Which component handles orchestration and memory?

A) Triton Inference Server with custom conversation storage
B) NeMo Framework with built-in memory backends
C) NeMo Agent Toolkit + AIQ Toolkit for memory providers and connectors
D) NIM with Redis for cross-session state

**Answer: C**

- NAT handles agent orchestration; AIQ provides memory and enterprise connectors
- A is wrong: Triton is serving only, not orchestration or memory
- B is wrong: NeMo Framework trains models; it's not an agent layer
- D is wrong: NIM is containerized inference; it's not an orchestration or memory layer

---

### Question 4

Your model needs to enforce content policies (blocking certain topics without fine-tuning). What's the right approach?

A) Fine-tune the model with NeMo Aligner to learn policy constraints
B) Implement custom guardrails in NIM before inference
C) Use NeMo Guardrails + Colang to add safety gates without modifying the model
D) Deploy the model with Triton and add safety rules to the request handler

**Answer: C**

- Guardrails + Colang is built for this exact use case
- A is wrong: Aligner is for RLHF, not policy enforcement
- B is wrong: NIM is just containerization; safety belongs in Guardrails
- D is wrong: Triton is serving, not a safety framework

---

### Question 5

You're deploying three different models (chat, embedding, code) to production and need auto-batching, concurrent request handling, and performance metrics per model. Which infrastructure component is essential?

A) NeMo Framework with multi-model serving
B) NIM instances for each model with a custom load balancer
C) Triton Inference Server with Model Analyzer for optimization
D) AIQ Toolkit with built-in serving capabilities

**Answer: C**

- Triton is the serving platform for multiple models with auto-batching and profiling
- A is wrong: NeMo Framework trains; it doesn't serve at scale
- B is wrong: Running separate NIM instances requires manual load balancing; Triton handles this
- D is wrong: AIQ is for agents and memory, not multi-model serving infrastructure

---

## Final Exam Checklist

Before submitting:

- Did I pick a tool that actually exists in the scenario description?
- Does the tool do EXACTLY what the question asks (not just related work)?
- Is there a more specific NVIDIA tool for this job?
- Could this be a trap about what a tool does NOT do?

Good luck. You've got this.

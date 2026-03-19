# Evaluation and Tuning Techniques (Domain 3)

## Overview

Evaluating agentic AI systems requires metrics across three layers: **inference performance**, **RAG quality**, and **agent behavior**. This domain covers how to measure, benchmark, and optimize systems in production.

---

## Part 1: Inference Performance Evaluation

### Key Metrics

| Metric | Definition | Why It Matters | Trade-off |
|--------|-----------|----------------|-----------|
| **Throughput** | Requests/tokens processed per second | Revenue and cost efficiency | Usually opposes latency |
| **Latency (P50/P99)** | Time from request to response | User experience | Higher throughput = higher latency |
| **Memory Utilization** | GPU/CPU memory consumption | Cost per concurrent user | More optimization = higher compute cost |
| **Token Generation Speed** | Tokens/second generated | Real-time interaction quality | Depends on batch size |

### Latency vs Throughput Trade-off

**Why they oppose each other:**
- **High throughput**: Batch multiple requests together → longer wait time for individual requests
- **Low latency**: Serve requests immediately → lower GPU utilization, fewer concurrent requests

**Production Decision:**
- **Interactive agents** (chat): Optimize for P99 latency (< 500ms)
- **Batch processing agents** (report generation): Optimize for throughput
- **Hybrid**: Use dynamic batching to balance both

### Measuring Inference Quality

#### TensorRT-LLM Optimizations (NVIDIA-specific)

| Optimization | Metric | Typical Improvement |
|---|---|---|
| **KV Cache Optimization** | Memory usage | -40% to -60% |
| **Paged Attention** | Max batch size | 2-4x increase |
| **In-Flight Batching** | Throughput | 3-10x improvement |
| **Tensor Parallelism** | Model size support | Scale to 8+ GPUs |
| **Quantization (INT8/FP8)** | Latency | -30% to -50% |
| **Custom CUDA Kernels** | Overall speedup | 2-3x for attention ops |

#### Dynamic Batching Configuration

```
Dynamic Batching = serving requests that arrive within a time window together

Configuration trade-offs:
- Max batch size: Larger = higher throughput, higher latency
- Max wait time: Longer = better batching, higher latency for early arrivals
- Preferred batch size: Hint to optimizer
```

**Tuning approach:**
1. Measure baseline latency (no batching)
2. Find "sweet spot" batch size for your GPU (usually 16-64 for LLMs)
3. Set max wait time = max acceptable latency increase
4. Monitor actual batch distribution (should match sweet spot)

---

## Part 2: RAG System Evaluation

### RAGAS Framework (Modern Standard)

RAGAS is the standard framework for evaluating RAG systems. It provides four key metrics:

| Metric | Measures | Formula | Target |
|--------|----------|---------|--------|
| **Faithfulness** | Does response match the context? | Fraction of claims grounded in context | 0.7+ |
| **Answer Relevance** | Does response address the question? | Semantic similarity between Q & A | 0.8+ |
| **Context Relevance** | Does retrieved context matter? | Fraction of context needed for answer | 0.6+ |
| **Context Recall** | Is all needed information retrieved? | Fraction of gold context actually retrieved | 0.8+ |

### Vector Search Quality Metrics

| Metric | What It Measures | Calculation |
|--------|---|---|
| **Precision@K** | Of top K results, how many are relevant? | Relevant in top K / K |
| **Recall@K** | Of all relevant docs, how many in top K? | Relevant in top K / Total relevant |
| **nDCG (Normalized DCG)** | Ranking quality (best results ranked high) | Standard IR metric |
| **MRR (Mean Reciprocal Rank)** | Position of first relevant result | 1 / rank of first relevant |

**Quick interpretation:**
- High Precision, Low Recall = Missing relevant docs (retrieval too strict)
- Low Precision, High Recall = Too many irrelevant docs (noisy results)
- Goal: Balance both (typically Precision > 0.7, Recall > 0.8)

### Common RAG Issues & How to Evaluate

| Problem | Symptom | Evaluation Method | Fix |
|---------|---------|---|---|
| **Embedding Mismatch** | Irrelevant results after model update | Dimension check, embedding similarity test | Use same model for index + query |
| **Retrieval Gaps** | Important docs not in top-K | Recall@K metric, manual review | Adjust K, improve chunking, rerank |
| **Lost in Middle** | LLM ignores middle context | Faithfulness metric, trace attention | Use RAG with re-ranking (CRAG pattern) |
| **Hallucination in RAG** | False claims despite context | Faithfulness metric | Self-RAG pattern (verify against context) |

### Advanced RAG Patterns with Evaluation

| Pattern | When to Use | Key Metric |
|---------|---|---|
| **Self-RAG** | High accuracy needs | Faithfulness (should be near 1.0) |
| **CRAG** (Corrective RAG) | Unreliable retrieval | Context Relevance (evaluates result quality) |
| **Adaptive RAG** | Unknown document complexity | Context Recall (decides when to stop retrieving) |
| **Query Rewriting** | Poor initial retrieval | Precision@K (after rewrite) |

---

## Part 3: Agent Performance Evaluation

### Agent Success Metrics

| Metric | Definition | How to Measure |
|--------|-----------|---|
| **Task Success Rate** | % of tasks completed correctly | Manual review or automated checklist |
| **Cost per Task** | Total API + compute cost per task | Token counters + infra metrics |
| **Latency (end-to-end)** | Total time from input to output | Timestamp logging |
| **Hallucination Rate** | % of outputs containing false claims | Ground truth comparison |
| **Tool Usage Accuracy** | % correct tool selections | Compare selected vs optimal tool |

### Reasoning Pattern Comparison (for choosing patterns)

| Pattern | Speed | Accuracy | Cost | When to Use |
|---------|-------|----------|------|---|
| **Direct (no reasoning)** | Very fast | Low | Cheapest | Simple queries, well-trained models |
| **Chain-of-Thought (CoT)** | Slow | High | Moderate | Complex multi-step reasoning |
| **Tree-of-Thoughts (ToT)** | Very slow | Very high | Expensive | Hard problems, low error tolerance |
| **ReAct** | Slow | High | Moderate | Multi-step with tool usage |
| **Self-Reflection** | Slow | Very high | Expensive | Quality critical, user-facing |

**Exam insight:** Pattern selection is about **task complexity vs budget**, not just accuracy.

### Benchmarking Agentic Systems

#### Setup
1. Create test dataset (100+ examples with ground truth)
2. Run agent on dataset
3. Collect metrics: latency, cost, success rate, hallucinations
4. Compare against baseline

#### Metrics to Track
- **Baseline latency** (no agent, just LLM)
- **Agent overhead** (difference due to reasoning + tools)
- **Tool accuracy** (correct tool selection %)
- **End-to-end success** (final answer correct?)

---

## Part 4: Production Tuning

### Latency Optimization Path

```
1. Baseline measurement (measure everything first)
2. Identify bottleneck (profiling tools)
   - LLM generation? → Use TensorRT-LLM, lower temperature
   - Tool calling? → Cache tool outputs, parallelize tools
   - Retrieval? → Optimize embeddings, reduce documents
3. Apply optimization
4. Measure improvement
5. Repeat until acceptable
```

### Cost Optimization Path

```
1. Measure cost breakdown
   - LLM tokens used?
   - API calls (NIMs, external tools)?
   - Compute (GPU hours)?
2. Identify highest cost component
3. Optimize:
   - LLM: Use smaller models, caching, prompt optimization
   - Tools: Batch calls, cache results
   - Compute: Better batching, resource sharing
4. Measure cost improvement
```

### Multi-GPU Scaling Metrics

| Metric | Good Target | Warning Sign |
|--------|---|---|
| **GPU Utilization** | 80-95% | < 50% = underutilized, > 95% = bottleneck |
| **Scaling Efficiency** | > 80% | < 60% = communication overhead too high |
| **Memory Distribution** | Balanced | > 20% imbalance = one GPU overloaded |

---

## Part 5: Monitoring & Continuous Improvement

### Key Metrics to Dashboard

1. **Service Health**
   - Request success rate (should be > 99%)
   - Error rate by type (timeouts, OOM, etc.)
   - P99 latency trend

2. **Quality Metrics**
   - Hallucination rate (detected via post-processing)
   - Tool accuracy (tracked per tool)
   - User satisfaction (if available)

3. **Resource Metrics**
   - GPU utilization and temperature
   - Memory pressure
   - Cost per request trend

4. **Business Metrics**
   - Requests per second
   - Tasks completed
   - Cost efficiency (cost per successful task)

### When to Retune

**Retune if:**
- Latency increases > 10% (indicates capacity issue)
- Error rate increases > 1% (indicates model drift or bug)
- Cost per task increases > 20% (indicates inefficiency)
- Hallucination rate increases (indicates model degradation)

---

## Exam-Focused Summary

**Domain 3 tests:**
- ✅ Understanding latency vs throughput trade-offs
- ✅ Choosing evaluation metrics for different use cases
- ✅ Using RAGAS framework for RAG evaluation
- ✅ Reasoning pattern selection (accuracy vs cost)
- ✅ Identifying bottlenecks and optimization strategies
- ✅ Production monitoring and alerting

**Key exam patterns:**
- "Which metric for real-time agent?" → P99 latency
- "RAG returns irrelevant results" → Check embedding model mismatch, then precision@K
- "Agent is too slow" → Profile first, then optimize highest-cost component
- "How to compare patterns?" → Look at accuracy/cost ratio for your use case

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

RAGAS is the standard framework for evaluating RAG systems. It provides metrics across two dimensions: **Retrieval** (are we getting the right documents?) and **Generation** (given context, is the LLM generating good answers?).

#### Retrieval Metrics (Evaluating the Retriever)

| Metric | Question It Answers | Ideal Range |
|--------|---|---|
| **Context Precision** | Of retrieved docs, how many are relevant? (Ranking quality?) | 0.7+ |
| **Context Recall** | Did we retrieve ALL relevant docs? (Missing anything?) | 0.8+ |
| **Context Relevance** | Is the retrieved context holistically useful for this query? | 0.6+ |

**Context Precision:** Of all context chunks retrieved, how many were actually relevant? Measures signal-to-noise in your retrieval. If you retrieve 10 chunks but only 3 are useful, precision is low. Also considers ranking - relevant chunks appearing higher score better. High precision = not polluting LLM context window with noise.

**Context Recall:** Of all relevant information that *exists* in your knowledge base, how much did the retriever find? If ground truth answer requires facts A, B, and C but you only retrieved A and B, recall is incomplete. Typically requires gold reference answers.

**Context Relevance:** How relevant is the retrieved context *as a whole* to the user's query? More holistic than precision - looks at whether passages are on-topic and useful for the specific question, not just counting relevant vs irrelevant chunks.

#### Generation Metrics (Evaluating the LLM Output)

| Metric | Question It Answers | Ideal Range |
|--------|---|---|
| **Faithfulness** | Does answer stick to context or hallucinate? | 0.7+ |
| **Answer Relevance** | Does answer actually address the question? | 0.8+ |
| **Answer Correctness** | Is answer factually correct vs ground truth? | 0.8+ |

**Faithfulness:** Does the generated answer **stick to the retrieved context**, or did the LLM hallucinate? Arguably the most critical RAG metric. Measures whether every claim in the answer can be traced back to provided context. If LLM invents facts not in documents, faithfulness drops. Score of 1.0 = zero hallucination vs context.

**Answer Relevance:** Does the generated answer actually address the user's question? An answer could be perfectly faithful to context but completely miss what was asked. Penalizes answers that are incomplete, vague, or off-topic - even if factually correct. Typically evaluated by checking if answer could regenerate the original question.

**Answer Correctness:** Is the generated answer **factually correct** when compared against ground truth? Combines semantic similarity and factual overlap with known correct answer. Unlike faithfulness (checks against *context*), correctness checks against *expected answer*. Requires labeled ground truth data.

#### RAGAS Exam Trap

High Faithfulness + Low Answer Relevance = Grounded answer that doesn't address the question
Low Faithfulness + High Answer Relevance = Addresses question but hallucinating!

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

## Part 6: Regularization and Preventing Overfitting

Regularization techniques constrain models during training so they generalize better to unseen data. Critical for production agents that must handle novel scenarios.

### Understanding Overfitting

**Overfitting** occurs when a model learns the training data *too well* - it memorizes specific examples including their noise and quirks, instead of learning underlying patterns that generalize to new data.

**Clear sign:** Large gap between training and validation performance
```
Example: Training accuracy 99% → Validation accuracy 70% = OVERFITTING
Training loss decreasing → Validation loss increasing = OVERFITTING
```

**Common causes:**
- Too little training data (fewer examples make memorization easier)
- Model too complex relative to data (too much capacity)
- Training for too many epochs (time to shift from learning to memorizing)
- Noisy or mislabeled data (model fits errors as patterns)

### Regularization Techniques

| Technique | How It Works | Best For | Trade-off |
| --- | --- | --- | --- |
| **Dropout** | Randomly deactivate neurons during training | General regularization | Can underfit if too aggressive |
| **L1 Regularization** | Penalize absolute weight values (sparsity) | Feature selection, sparse models | Unstable with correlated features |
| **L2 Regularization** | Penalize squared weights (shrink towards zero) | Discouraging large weights | Doesn't eliminate features |
| **Elastic Net** | Combine L1 + L2 | Balance between selection & shrinking | More hyperparameters to tune |
| **Early Stopping** | Stop when validation loss rises | Simple, effective general approach | Requires validation monitoring |
| **Data Augmentation** | Expand training set with transformations | Limited training data | Domain-specific augmentations needed |
| **Batch Normalization** | Normalize layer inputs | Speed up training (mild regularization) | Slight overhead at inference |
| **Label Smoothing** | Soften training labels (e.g., 90% class A, 10% others) | Prevent overconfidence | May reduce accuracy on confident decisions |

### Deep Dive: Dropout

Dropout randomly "turns off" a percentage of neurons during each training step. Example: 50% dropout = half the neurons deactivated randomly on each forward pass.

**Why this helps:**
- Prevents co-dependent neurons from memorizing training patterns
- Forces network to learn redundant representations
- Multiple pathways to correct output = more robust on unseen data

**Practical details:**
- Only active during training - all neurons used at inference
- Common dropout rates: 0.2-0.5
- Too little: doesn't help with overfitting
- Too much: prevents learning (underfitting)

**Analogy:** Like randomly sending team members home on different days, forcing everyone to learn all skills. Result: a stronger, more versatile team.

### Deep Dive: L1 vs L2 Regularization

Both add penalties to the loss function during training:

**L1 Regularization (Lasso):**
- Penalty = sum of absolute weight values
- Effect: Pushes many weights all the way to zero
- Use when: Only handful of features truly matter
- Advantage: Automatic feature selection
- Disadvantage: Unstable with correlated features

**L2 Regularization (Ridge):**
- Penalty = sum of squared weight values
- Effect: Shrinks all weights toward smaller values
- Use when: Most features contribute somewhat
- Advantage: Most commonly used, built into many optimizers as "weight decay"
- Disadvantage: Doesn't eliminate features

**Elastic Net:** Combines L1 and L2 - balance controlled by mixing parameter.

### Early Stopping: Simple and Effective

Monitor validation loss during training. Stop when validation loss stops improving (even while training loss keeps decreasing).

**Why it works:** Model never gets chance to fully memorize training data.

**Key metrics:**
- Validation loss trend (must improve)
- Patience parameter (how many steps without improvement before stopping)
- Checkpoint best model (keep weights at lowest validation loss)

### Data Augmentation

Artificially expand training set by applying transformations:
- Images: Rotation, flipping, cropping, brightness adjustment
- Audio: Adding noise, time-stretching
- Text: Paraphrasing, synonym replacement
- Time series: Jittering, scaling, warping

**Effect:** More diverse training data makes memorization harder, forces learning general patterns.

### Other Regularization Techniques

**Batch Normalization:** Normalizes inputs to each layer for consistent mean/variance. Has mild regularizing effect since it introduces small noise (depends on mini-batch statistics).

**Label Smoothing:** Soften training labels instead of "100% cat" say "90% cat, 10% spread across other classes." Prevents overconfidence, improves generalization.

**Weight Tying:** Force different network parts to share weights, reducing total parameters. Common in language models (input embedding and output layers share weights).

**Noise Injection:** Deliberately add random noise to inputs, weights, or gradients during training. Dropout is actually a special case - multiplicative Bernoulli noise in activations.

### Regularization in Agent Training

For agentic systems, regularization is critical for:
- **Domain adaptation:** Agent trained on one domain must generalize to similar domains
- **Prompt sensitivity:** Agent shouldn't overfit to exact prompt wording
- **Tool selection:** Agent learns general tool patterns, not memorize training examples
- **Few-shot learning:** Critical for learning from limited examples



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

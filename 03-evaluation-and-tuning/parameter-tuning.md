# Tuning Model Parameters (Objective 3.4)

Practical parameter tuning for accuracy, latency, and cost trade-offs. The existing eval file covers inference-level metrics and regularization. This file focuses on the parameters you actually tune in production agent systems.

---

## LLM Generation Parameters

These are the knobs you turn when configuring the LLM within your agent.

### Temperature

Controls randomness in token selection.

| Temperature | Behavior | Use For |
|---|---|---|
| 0.0 | Deterministic (always picks most likely token) | Tool calling, classification, factual Q&A |
| 0.1-0.3 | Very focused, consistent | Production agents, structured output |
| 0.5-0.7 | Balanced creativity and consistency | General conversation, summarization |
| 0.8-1.0 | Creative, diverse outputs | Brainstorming, creative writing |
| >1.0 | Very random, potentially incoherent | Almost never useful for agents |

For agents, temperature 0.0-0.3 is typical. You want consistent, reliable tool selection and reasoning. Higher temperature introduces unpredictable behavior.

Exam trap: "Temperature 0" doesn't mean the model always gives the same answer (due to floating point, batching effects), but it's as deterministic as possible.

### Top-p (Nucleus Sampling)

Instead of considering all possible next tokens, only consider tokens whose cumulative probability reaches p.

```
top_p = 0.9: Consider tokens until you've covered 90% of the probability mass
top_p = 0.1: Only consider the very top tokens (very focused)
top_p = 1.0: Consider all tokens (no filtering)
```

In practice, either tune temperature OR top_p, not both. Most agent configurations:
- Temperature 0, top_p 1.0 (deterministic)
- Or temperature 0.7, top_p 0.9 (balanced)

### Top-k

Only consider the top k most likely tokens at each step.

```
top_k = 1:  Greedy decoding (always pick the most likely token)
top_k = 10: Choose from top 10 tokens
top_k = 50: Moderate diversity
```

Less commonly tuned than temperature and top_p for agent use cases.

### Max Tokens

Maximum length of the generated response.

| Setting | Trade-off |
|---|---|
| Too low (e.g., 100) | Agent response gets cut off mid-sentence |
| Too high (e.g., 4096) | Agent might ramble, uses more tokens (cost), higher latency |
| Just right | Depends on task. Short for classification (50), medium for Q&A (500), long for reports (2000) |

For agents with tool calling, you need enough tokens for the tool call JSON plus the response. Typically 1024-2048 is safe.

### Stop Sequences

Tokens that tell the model to stop generating.

```python
stop_sequences = ["\nObservation:", "\nHuman:", "FINAL ANSWER:"]
```

Critical for ReAct agents: you want the model to stop after generating an Action, not continue hallucinating the Observation. The stop sequence `"\nObservation:"` stops generation so the tool can actually execute.

### Frequency and Presence Penalties

| Parameter | Effect |
|---|---|
| **Frequency penalty** (0-2) | Reduces repetition of tokens already used (proportional to count) |
| **Presence penalty** (0-2) | Reduces use of any token that appeared at all (binary) |

For agents: usually leave at 0. These are more relevant for creative writing. Agents need to be able to repeat technical terms and structured formats.

---

## Accuracy vs Latency vs Cost Trade-offs

### The Fundamental Triangle

```
        Accuracy
       /        \
      /          \
   Cost -------- Latency
```

You can optimize for two at the expense of the third:
- **High accuracy + low latency** = expensive (big model, lots of GPUs)
- **High accuracy + low cost** = slow (big model, fewer GPUs, longer batching)
- **Low latency + low cost** = less accurate (small model)

### Quantization Trade-offs

Reduce model precision to speed up inference at the cost of some accuracy.

| Precision | Memory | Speed | Accuracy Impact | NVIDIA Support |
|---|---|---|---|---|
| FP32 | Baseline | Baseline | None | All GPUs |
| FP16 | 50% less | ~2x faster | Minimal | All modern GPUs |
| BF16 | 50% less | ~2x faster | Minimal | A100, H100 |
| INT8 | 75% less | ~3x faster | Small (1-2% drop) | TensorRT-LLM |
| FP8 | 75% less | ~3x faster | Small | H100 only |
| INT4 (AWQ/GPTQ) | 87% less | ~4x faster | Moderate (2-5% drop) | TensorRT-LLM |

Decision framework:
1. Start with FP16/BF16 (nearly free accuracy-wise)
2. Try INT8 and benchmark. If accuracy holds, use it.
3. INT4 only if you need to fit the model on fewer GPUs and can tolerate the accuracy drop

NVIDIA NIM automatically applies TensorRT-LLM optimizations including quantization.

### Model Size Selection

| Need | Model Size | Example |
|---|---|---|
| Classification, routing | 7-8B | Llama 3.1 8B |
| General agent tasks | 13-34B | Mixtral 8x7B |
| Complex reasoning, coding | 70B+ | Llama 3.1 70B, Nemotron |

Benchmark data should drive this choice, not assumptions. Sometimes a well-prompted 8B model beats a poorly-prompted 70B model.

---

## RAG Parameter Tuning

### Retrieval Parameters

| Parameter | Effect | Tuning Approach |
|---|---|---|
| **Top-K results** | How many chunks to retrieve | Start with 5, increase if recall is low, decrease if precision is low |
| **Chunk size** | Size of document chunks (tokens) | 256-1024 typical. Smaller = more precise, Larger = more context |
| **Chunk overlap** | Overlap between adjacent chunks | 10-20% of chunk size prevents cutting important context |
| **Similarity threshold** | Minimum relevance score | Set based on your data. Too high = misses results, too low = noise |
| **Reranking** | Re-score retrieved chunks with a cross-encoder | Almost always improves quality. Use NVIDIA NV-RerankQA |

### Embedding Model Selection

| Factor | Consideration |
|---|---|
| **Dimension** | Higher dimensions (1024+) = more expressive but more storage and compute |
| **Domain match** | General embeddings vs domain-fine-tuned. Domain-tuned usually wins. |
| **Multilingual** | If your data is multilingual, use a multilingual embedding model |
| **Speed** | Smaller embedding models are faster for real-time retrieval |

NVIDIA NV-Embed models are optimized for retrieval tasks and run efficiently on NIMs.

### Chunking Strategy Impact

```
Strategy A: 256 token chunks, no overlap
  Precision@5: 0.82  Recall@5: 0.65  → Missing relevant content

Strategy B: 512 token chunks, 50 token overlap
  Precision@5: 0.78  Recall@5: 0.79  → Better recall, slightly noisier

Strategy C: 512 token chunks, 50 overlap, with reranking
  Precision@5: 0.85  Recall@5: 0.79  → Best of both worlds
```

Reranking (using a cross-encoder model to re-score retrieved chunks) is one of the highest-impact RAG improvements.

---

## Agent-Level Parameter Tuning

### Max Iterations / Max Steps

How many ReAct loops the agent can take before stopping.

```python
agent = create_agent(
    llm=llm,
    tools=tools,
    max_iterations=10  # stop after 10 think-act-observe cycles
)
```

Too low: Agent gives up on complex tasks.
Too high: Agent gets stuck in loops, wastes tokens and time.

Tune based on your task complexity. Simple tasks need 2-3 steps, complex tasks might need 8-10.

### Tool Timeout

How long to wait for a tool call before giving up.

```python
tools = [
    Tool(name="search", func=search, timeout=5.0),     # 5 second timeout
    Tool(name="database", func=db_query, timeout=10.0), # 10 second timeout
]
```

### Retry Configuration

How many times to retry failed tool calls or LLM generations.

```python
retry_config = {
    "max_retries": 3,
    "backoff_factor": 2,          # exponential backoff
    "retry_on": ["timeout", "rate_limit", "server_error"],
    "no_retry_on": ["invalid_input", "not_found"]
}
```

---

## Systematic Tuning Methodology

### 1. Establish Baseline

Before tuning anything, measure your current performance:

```
Baseline (default params):
  Accuracy: 82%
  P50 Latency: 2.3s
  P99 Latency: 6.8s
  Cost/query: $0.006
  Tool accuracy: 85%
```

### 2. One Variable at a Time

Change one parameter, measure impact, then decide:

```
Experiment 1: Temperature 0 → 0.2
  Accuracy: 82% → 83% (+1%)
  Latency: unchanged
  → Keep: slight accuracy improvement, no cost

Experiment 2: Top-K retrieval 3 → 5
  Accuracy: 83% → 86% (+3%)
  Latency: 2.3s → 2.5s (+9%)
  → Keep: good accuracy gain, small latency increase

Experiment 3: Add reranking
  Accuracy: 86% → 89% (+3%)
  Latency: 2.5s → 3.1s (+24%)
  Cost: +$0.001/query
  → Keep: significant accuracy gain, acceptable cost

Experiment 4: Max iterations 5 → 10
  Accuracy: 89% → 90% (+1%)
  Latency: 3.1s → 3.8s (+23%)
  Cost: +$0.003/query
  → Reject: minimal accuracy gain, significant cost increase
```

Final tuned config: temperature=0.2, top_k=5, reranking=on, max_iterations=5.

### 3. Grid Search for Key Parameters

When you have a few important parameters to tune together:

```python
param_grid = {
    "temperature": [0, 0.1, 0.3],
    "top_k_retrieval": [3, 5, 10],
    "chunk_size": [256, 512],
    "reranking": [True, False]
}

# Run each combination on the benchmark
best_config = None
best_score = 0
for config in product(*param_grid.values()):
    results = run_benchmark(agent, test_set, config)
    score = results.accuracy * 0.6 + (1 - results.latency/10) * 0.2 + (1 - results.cost/0.01) * 0.2
    if score > best_score:
        best_config = config
        best_score = score
```

The scoring formula reflects your priorities (60% accuracy, 20% speed, 20% cost). Adjust weights based on your use case.

---

## Exam-Style Questions

**Q1: An agent sometimes selects the wrong tool because its reasoning varies between runs. What parameter should you adjust first?**
- A) Max tokens
- B) Temperature (lower it toward 0)
- C) Top-K retrieval
- D) Chunk size

**Answer: B** - Inconsistent tool selection is caused by randomness in generation. Lowering temperature toward 0 makes the agent more deterministic and consistent.

**Q2: After quantizing a model from FP16 to INT8, accuracy drops 3%. What should you do?**
- A) Accept the trade-off since INT8 is always better
- B) Benchmark the quantized model and decide if the latency/memory savings justify the accuracy drop
- C) Always use FP32
- D) Quantize further to INT4

**Answer: B** - Quantization is a trade-off. Benchmark the quantized model on your specific task. If 3% accuracy loss is acceptable for the speed/memory gain, use INT8. If not, stay with FP16.

**Q3: A RAG agent retrieves relevant documents but the final answer often misses important details. Which parameter should you tune?**
- A) LLM temperature
- B) Chunk size (try larger chunks to include more context)
- C) Stop sequences
- D) Frequency penalty

**Answer: B** - If retrieval is good but the answer misses details, chunks might be too small, cutting off important context. Larger chunks include more surrounding information.

**Q4: What is the correct approach to parameter tuning?**
- A) Change all parameters at once for maximum improvement
- B) Change one parameter at a time, measure impact, then decide
- C) Always use default parameters
- D) Copy parameters from a published paper

**Answer: B** - One variable at a time lets you isolate which changes help and which hurt. Changing multiple parameters simultaneously makes it impossible to attribute improvements.

---

## Key Takeaways

1. Temperature 0-0.3 for agents (deterministic tool selection and reasoning)
2. Quantization (FP16 → INT8 → INT4) trades accuracy for speed/memory. Always benchmark.
3. Model size selection should be driven by benchmarks, not assumptions
4. RAG tuning: chunk size, top-K, overlap, and reranking are the highest-impact parameters
5. Max iterations controls agent loop depth. Too low = gives up, too high = wasted tokens
6. Tune one variable at a time against your benchmark to isolate impact
7. The accuracy/latency/cost triangle means you always trade one for the others
8. NVIDIA NIM applies TensorRT-LLM optimizations automatically (quantization, batching, caching)

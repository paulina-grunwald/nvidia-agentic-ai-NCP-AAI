# Comparing Agent Performance Across Tasks and Datasets

How to systematically compare agent configurations, models, and strategies to make informed decisions.

---

## Why Comparison Matters

You often need to answer questions like:

- Is agent v2 actually better than v1?
- Should we use Llama 3.1 70B or Mixtral 8x7B?
- Is ReAct better than simple CoT for our use case?
- Does adding self-reflection justify the extra latency and cost?

Without systematic comparison, these become opinion-based decisions. With proper benchmarking, they become data-driven.

---

## Comparison Dimensions

When comparing agents, measure across multiple dimensions. An agent that's "better" on accuracy might be worse on cost.

| Dimension             | Metric                                         | Why It Matters                 |
| --------------------- | ---------------------------------------------- | ------------------------------ |
| **Accuracy**          | Task success rate, answer correctness          | Does it get the right answer?  |
| **Latency**           | P50, P95, P99 response times                   | Is it fast enough for users?   |
| **Cost**              | Tokens used, API calls, compute time           | Can we afford to run it?       |
| **Robustness**        | Performance on edge cases, adversarial inputs  | Does it break easily?          |
| **Tool usage**        | Correct tool selection rate, unnecessary calls | Does it use tools efficiently? |
| **Reasoning quality** | Trajectory score, step correctness             | Is the reasoning path logical? |

### The Spider Chart Approach

Visualize agent performance across all dimensions:

```
         Accuracy
            │
    Cost ───┼─── Latency
            │
       Robustness

Agent A: High accuracy, high cost, medium latency
Agent B: Medium accuracy, low cost, low latency

→ Agent A for quality-critical tasks
→ Agent B for high-volume, cost-sensitive tasks
```

---

## Cross-Task Comparison

### Performance Matrix

Run each agent configuration on every task category:

```
                    Simple Q&A   Comparison   Troubleshoot   Multi-step
Agent v1 (CoT)      92%          71%          80%            65%
Agent v2 (ReAct)    88%          82%          87%            78%
Agent v3 (ReAct+SR) 90%          85%          91%            83%

Latency (avg):
Agent v1             1.2s         2.1s         2.5s           3.8s
Agent v2             1.8s         3.2s         3.5s           5.1s
Agent v3             2.9s         5.0s         5.8s           8.2s
```

Reading this: v3 (ReAct + self-reflection) is most accurate everywhere but much slower. v1 wins on simple Q&A where CoT is enough. v2 is the best trade-off for complex tasks.

### Statistical Significance

Don't conclude "v2 is better" from a 2% difference on 50 test cases. That could be noise.

Rules of thumb:

- 100+ test cases per category for meaningful comparison
- 5%+ difference to be confident (with 100 cases)
- Run multiple times (3-5 runs) to check variance
- Use confidence intervals, not just averages

```python
from scipy import stats

# Compare accuracy of two agents
agent_a_scores = [1, 0, 1, 1, 0, 1, ...]  # 1=correct, 0=incorrect
agent_b_scores = [1, 1, 1, 0, 1, 1, ...]

# McNemar's test for paired binary outcomes
stat, p_value = stats.mcnemar(contingency_table)
if p_value < 0.05:
    print("Significant difference between agents")
else:
    print("Difference could be due to chance")
```

---

## Cross-Dataset Comparison

### Generalization Testing

An agent trained/optimized on one dataset should work on others:

```
                    Training Dataset   Held-out Test   New Domain
Agent (optimized)    91%               85%             72%
```

If performance drops sharply on new domains, the agent is overfitting to its training data. This connects to the regularization content in the existing eval file.

### Domain Transfer Matrix

```
              Trained On →
Tested On ↓   Finance    Tech Support   Legal
Finance       91%        68%            55%
Tech Support  62%        89%            51%
Legal         49%        53%            87%
```

Diagonal = in-domain performance. Off-diagonal = how well the agent transfers. Low off-diagonal means the agent is too specialized.

---

## Model Comparison

### LLM Selection Benchmarking

When choosing which LLM to use as the agent's backbone:

```python
models_to_test = [
    {"name": "Llama-3.1-8B", "nim_endpoint": "..."},
    {"name": "Llama-3.1-70B", "nim_endpoint": "..."},
    {"name": "Mixtral-8x7B", "nim_endpoint": "..."},
    {"name": "Nemotron-70B", "nim_endpoint": "..."},
]

for model in models_to_test:
    agent = create_agent(llm=model["nim_endpoint"])
    results = run_benchmark(agent, test_dataset)
    print(f"{model['name']}: accuracy={results.accuracy}, latency={results.p50_ms}ms, cost=${results.cost_per_query}")
```

Typical trade-offs:

| Model Size    | Accuracy | Latency | Cost           | When to Use                    |
| ------------- | -------- | ------- | -------------- | ------------------------------ |
| 8B            | Lower    | Fast    | Cheap          | Simple routing, classification |
| 70B           | Higher   | Slower  | More expensive | Complex reasoning, generation  |
| MoE (Mixtral) | Good     | Medium  | Medium         | Balanced workloads             |

### Model Routing Strategy

Use comparison data to route queries to appropriate models:

```python
def route_to_model(query):
    complexity = classify_complexity(query)
    if complexity == "simple":
        return "llama-3.1-8b"   # fast and cheap
    elif complexity == "moderate":
        return "mixtral-8x7b"   # balanced
    else:
        return "llama-3.1-70b"  # maximum accuracy
```

This is a real production pattern. Benchmark data tells you which model handles which difficulty level best.

---

## Strategy Comparison

### Reasoning Pattern Benchmarking

The existing eval file covers pattern trade-offs at a high level. Here's how to do systematic comparison:

```
Test set: 200 queries, 4 categories

Strategy: Direct (no reasoning)
  Simple:  95%  0.8s  $0.001
  Compare: 45%  0.9s  $0.001
  Debug:   30%  0.8s  $0.001
  Multi:   20%  0.9s  $0.001

Strategy: CoT
  Simple:  93%  1.5s  $0.003
  Compare: 72%  2.8s  $0.005
  Debug:   68%  2.5s  $0.004
  Multi:   55%  3.2s  $0.006

Strategy: ReAct
  Simple:  90%  2.0s  $0.008
  Compare: 83%  4.5s  $0.015
  Debug:   85%  4.0s  $0.012
  Multi:   78%  6.0s  $0.020

Strategy: ReAct + Self-Reflection
  Simple:  91%  3.5s  $0.015
  Compare: 87%  7.0s  $0.028
  Debug:   90%  6.5s  $0.025
  Multi:   84%  9.0s  $0.035
```

Insights from this data:

- Direct is great for simple queries (95% accuracy at minimal cost)
- CoT adds little value for simple queries but helps a lot for comparisons
- ReAct is necessary for debugging (needs tools) and multi-step tasks
- Self-reflection adds 2-5% accuracy but doubles cost and latency

Decision: use Direct for simple, CoT for moderate, ReAct for complex. Only add self-reflection for high-stakes queries.

---

## Comparison Reporting

### Comparison Dashboard Template

```
═══════════════════════════════════════════════
Agent Comparison: v2.2 vs v2.3
Test Set: customer_support_v3 (200 cases)
Date: 2025-03-22
═══════════════════════════════════════════════

                        v2.2        v2.3        Δ
Overall Accuracy:       84%         87%         +3% ✅
P50 Latency:           2.1s        2.3s        +10% ⚠
P99 Latency:           5.8s        4.9s        -16% ✅
Cost per Query:        $0.005      $0.004      -20% ✅
Tool Accuracy:         88%         93%         +5% ✅

By Category:
  order_lookup:        90%→92%     (+2%)
  troubleshooting:     82%→88%     (+6%) ✅ Big improvement
  billing:             76%→79%     (+3%)
  recommendations:     85%→84%     (-1%) ⚠ Slight regression

Recommendation: Deploy v2.3
- Net positive on accuracy, cost, and P99 latency
- Minor regression on recommendations (investigate)
- P50 latency increase is acceptable given accuracy gains
═══════════════════════════════════════════════
```

---

## Exam-Style Questions

**Q1: Two agent configurations are compared on 50 test cases. Agent A scores 82%, Agent B scores 85%. Can you confidently say B is better?**

- A) Yes, 3% is a meaningful difference
- B) No, 50 cases is too small for a 3% difference to be statistically significant
- C) Yes, if B is newer it must be better
- D) No, accuracy doesn't matter

**Answer: B** - With only 50 cases, a 3% difference (about 1.5 cases) could easily be noise. Need 100+ cases and ideally statistical significance testing.

**Q2: An agent scores 91% accuracy on its training domain but 72% on a new domain. What does this indicate?**

- A) The agent is very good
- B) The new domain is too hard
- C) The agent may be overfitting to its training domain and not generalizing well
- D) The evaluation is wrong

**Answer: C** - A large drop between in-domain and out-of-domain performance suggests overfitting. The agent learned patterns specific to its training data rather than general skills.

**Q3: Benchmarking shows a small model (8B) achieves 95% accuracy on simple queries vs 97% for a large model (70B). Which should you use for simple queries?**

- A) Always the 70B for maximum accuracy
- B) The 8B since it's nearly as accurate but faster and cheaper
- C) Neither, use a rule-based system
- D) Always the smallest available model

**Answer: B** - For simple queries where the small model is nearly as accurate (95% vs 97%), the cost and latency savings of the 8B model make it the better choice. This is the model routing pattern.

**Q4: What is the primary purpose of running evaluation benchmarks multiple times (3-5 runs)?**

- A) To increase the dataset size
- B) To check for variance and ensure results are stable, not fluky
- C) To warm up the GPU
- D) To find the best random seed

**Answer: B** - Multiple runs reveal variance. If results swing wildly between runs, a single benchmark isn't reliable. Stable results across runs give confidence.

---

## Key Takeaways

1. Compare across multiple dimensions: accuracy, latency, cost, robustness, tool usage
2. Use 100+ test cases per category for statistically meaningful comparisons
3. Cross-task comparison reveals which strategies work best for which query types
4. Cross-dataset comparison tests generalization (watch for overfitting)
5. Model routing uses benchmark data to send queries to the right-sized model
6. Strategy selection (Direct/CoT/ReAct/ReAct+SR) should be based on empirical comparison, not intuition
7. Always report by category, not just overall, to catch regressions in specific areas
8. Statistical significance matters: small differences on small datasets aren't reliable

# Evaluation Pipelines and Task Benchmarks

How to build systematic evaluation pipelines that measure agent performance reliably and repeatedly.

---

## Why You Need an Evaluation Pipeline

Running a few test queries manually isn't enough. A proper eval pipeline:

- Runs automatically (CI/CD integrated)
- Tests against a fixed dataset so results are comparable over time
- Measures multiple dimensions (accuracy, latency, cost, tool usage)
- Catches regressions before they hit production

Without a pipeline, you're guessing. With one, you have data.

---

## Anatomy of an Agent Evaluation Pipeline

```
Test Dataset          Agent Under Test         Metrics Collection
(inputs + expected)   (your agent)             (automated scoring)
       ↓                    ↓                         ↓
┌─────────────┐     ┌──────────────┐         ┌──────────────┐
│ 200+ test   │ →   │ Run agent on │ →       │ Score each   │ → Results
│ cases with  │     │ each test    │         │ response     │   Report
│ ground truth│     │ case         │         │ (auto + LLM) │
└─────────────┘     └──────────────┘         └──────────────┘
                          ↓
                   ┌──────────────┐
                   │ Log traces:  │
                   │ - tool calls │
                   │ - reasoning  │
                   │ - latency    │
                   │ - tokens     │
                   └──────────────┘
```

### Step 1: Build the Test Dataset

A good test dataset has:

| Component           | Description                | Example                                     |
| ------------------- | -------------------------- | ------------------------------------------- |
| **Input**           | The user query/task        | "What's NVIDIA's latest quarterly revenue?" |
| **Expected output** | Ground truth answer        | "$35.1 billion in Q4 FY2025"                |
| **Expected tools**  | Which tools should be used | ["financial_data_api"]                      |
| **Category**        | Type of question           | "financial_lookup"                          |
| **Difficulty**      | Simple/medium/complex      | "medium"                                    |

```python
test_cases = [
    {
        "input": "What's NVIDIA's latest quarterly revenue?",
        "expected_output": "$35.1 billion in Q4 FY2025",
        "expected_tools": ["financial_data_api"],
        "category": "financial_lookup",
        "difficulty": "simple"
    },
    {
        "input": "Compare NVIDIA and AMD GPU performance for LLM inference",
        "expected_output": "...",  # detailed comparison
        "expected_tools": ["search_benchmarks", "product_database"],
        "category": "comparison",
        "difficulty": "complex"
    },
    # ... 200+ cases
]
```

Aim for at least 100-200 test cases. Cover edge cases, different categories, and varying difficulty levels.

### Step 2: Run the Agent

Execute the agent on each test case and capture everything:

```python
results = []
for case in test_cases:
    start_time = time.time()

    # Run agent with full trace logging
    response = agent.run(
        case["input"],
        trace=True  # capture reasoning steps, tool calls
    )

    elapsed = time.time() - start_time

    results.append({
        "input": case["input"],
        "expected": case["expected_output"],
        "actual": response.answer,
        "tools_used": response.tools_called,
        "reasoning_steps": response.trace,
        "latency_ms": elapsed * 1000,
        "tokens_used": response.token_count,
        "cost": response.estimated_cost
    })
```

### Step 3: Score with Multiple Methods

**Automated scoring** (fast, objective):

```python
def exact_match(expected, actual):
    return expected.lower().strip() == actual.lower().strip()

def contains_key_facts(expected, actual, key_facts):
    return all(fact.lower() in actual.lower() for fact in key_facts)

def tool_accuracy(expected_tools, actual_tools):
    return set(expected_tools) == set(actual_tools)
```

**LLM-as-Judge scoring** (nuanced, handles open-ended answers):

```python
judge_prompt = """
Rate this agent response on 1-5 for each dimension:

Query: {input}
Agent Response: {actual}
Reference Answer: {expected}

Accuracy (1-5): Is the response factually correct?
Completeness (1-5): Does it address all parts of the query?
Groundedness (1-5): Are claims supported by evidence?
Helpfulness (1-5): Would a user find this useful?

Return scores as JSON.
"""
```

**RAGAS scoring** (for RAG-based agents):

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# Evaluate RAG quality
ragas_results = evaluate(
    dataset=test_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
```

### Step 4: Generate Reports

```
═══════════════════════════════════════════════
Agent Evaluation Report - v2.3 - 2025-03-22
═══════════════════════════════════════════════

Overall Metrics:
  Task Success Rate:  87% (174/200)
  Avg Latency:        2.3s
  Avg Cost/Query:     $0.004
  Hallucination Rate: 4.5%

By Category:
  financial_lookup:   92% accuracy, 1.8s avg
  comparison:         78% accuracy, 3.9s avg
  troubleshooting:    85% accuracy, 2.7s avg

Tool Usage:
  Correct tool selected: 91%
  Unnecessary tool calls: 8%
  Missing tool calls: 3%

vs Previous Version (v2.2):
  Accuracy: +3% (84% → 87%)
  Latency:  -12% (2.6s → 2.3s)
  Cost:     -8% ($0.0043 → $0.004)

Regressions:
  ⚠ "comparison" category dropped 2% accuracy
  ⚠ P99 latency increased for complex queries
═══════════════════════════════════════════════
```

---

## NVIDIA Evaluation Tools

### NVIDIA Agent Intelligence Toolkit (AIQ)

The AIQ Toolkit includes evaluation capabilities for agent workflows:

- **Profiling**: Measure latency at each node in the agent graph
- **Trace analysis**: See exactly which tools were called and what reasoning was used
- **Metric collection**: Automated accuracy, latency, and cost metrics
- **Comparison**: A/B test different agent configurations

### NVIDIA NeMo Evaluator

Part of the NeMo ecosystem for evaluating model quality:

- Runs standard benchmarks (MMLU, HumanEval, etc.) against NIM endpoints
- Custom evaluation datasets
- Automated scoring with reference answers
- Integration with NeMo training pipelines for feedback loops

### Triton Performance Analyzer

For inference-level evaluation (not agent-level):

```bash
# Measure inference performance of a model on Triton
perf_analyzer -m llama-3-70b \
    --input-data test_prompts.json \
    --measurement-interval 10000 \
    --concurrency-range 1:32
```

Outputs: throughput, latency percentiles (P50, P90, P99), GPU utilization.

---

## Task Benchmarks for Agents

### Standard Benchmarks

| Benchmark      | What It Tests                       | Agent Relevance              |
| -------------- | ----------------------------------- | ---------------------------- |
| **MMLU**       | General knowledge (multiple choice) | Baseline LLM quality         |
| **HumanEval**  | Code generation correctness         | Code agent accuracy          |
| **GSM8K**      | Math reasoning                      | Reasoning agent quality      |
| **MINT-Bench** | Multi-turn interaction              | Conversational agent quality |
| **AgentBench** | Agent tool use and planning         | Direct agent evaluation      |
| **ToolBench**  | API/tool usage across domains       | Tool selection accuracy      |
| **SWE-bench**  | Software engineering tasks          | Coding agent capability      |

### Building Custom Task Benchmarks

For your specific use case, standard benchmarks aren't enough. Build domain-specific ones:

```python
# Example: Customer support agent benchmark
benchmark = {
    "name": "customer_support_v1",
    "categories": [
        {"name": "order_lookup", "count": 50, "difficulty": "simple"},
        {"name": "troubleshooting", "count": 40, "difficulty": "medium"},
        {"name": "billing_dispute", "count": 30, "difficulty": "complex"},
        {"name": "product_recommendation", "count": 30, "difficulty": "medium"},
        {"name": "edge_cases", "count": 20, "difficulty": "hard"},
    ],
    "total_cases": 170,
    "metrics": ["accuracy", "latency", "tool_accuracy", "customer_satisfaction"],
    "passing_criteria": {
        "accuracy": 0.85,
        "latency_p99_ms": 5000,
        "tool_accuracy": 0.90
    }
}
```

### Benchmark Best Practices

1. **Diverse coverage**: Include easy, medium, and hard cases
2. **Edge cases**: Ambiguous queries, missing data, out-of-scope requests
3. **Adversarial cases**: Prompts designed to trigger hallucination or unsafe responses
4. **Version the benchmark**: As you add cases, keep old versions for comparison
5. **Separate dev/test**: Don't optimize prompts against the same set you evaluate on
6. **Regular updates**: Add new cases from production failures

---

## CI/CD Integration

Run eval pipeline on every agent change:

```yaml
# .github/workflows/agent-eval.yml
name: Agent Evaluation
on:
  pull_request:
    paths: ["agent/**", "prompts/**", "tools/**"]

jobs:
  evaluate:
    runs-on: gpu-runner
    steps:
      - name: Run evaluation pipeline
        run: python eval/run_pipeline.py --dataset benchmarks/v3.json

      - name: Check pass criteria
        run: python eval/check_thresholds.py --min-accuracy 0.85 --max-latency-p99 5000

      - name: Post results to PR
        run: python eval/post_report.py
```

This catches regressions before they're merged. If accuracy drops or latency spikes, the PR is flagged.

---

## Exam-Style Questions

**Q1: What is the minimum recommended size for an agent evaluation test dataset?**

- A) 10-20 cases
- B) 100-200+ cases with ground truth
- C) 5 cases per category
- D) As many as possible, quality doesn't matter

**Answer: B** - 100-200+ cases with ground truth, covering different categories and difficulty levels, gives statistically meaningful results.

**Q2: An evaluation pipeline shows 87% overall accuracy but 78% on "comparison" queries. What should you investigate first?**

- A) Increase the overall dataset size
- B) Analyze failure cases in the "comparison" category to find patterns
- C) Reduce the number of comparison queries in the benchmark
- D) Switch to a larger model

**Answer: B** - Analyze specific failures in the weak category to understand why the agent struggles (wrong tool? bad reasoning? missing data?) before making changes.

**Q3: Which NVIDIA tool would you use to measure inference-level latency percentiles (P50, P99) for a model deployed on Triton?**

- A) NeMo Evaluator
- B) Triton Performance Analyzer
- C) AIQ Toolkit
- D) NeMo Guardrails

**Answer: B** - Triton Performance Analyzer (perf_analyzer) specifically measures inference performance metrics like latency percentiles and throughput.

**Q4: Why should evaluation datasets be versioned?**

- A) To reduce storage costs
- B) So results from different agent versions can be compared against the same benchmark
- C) To make the benchmark harder over time
- D) Version control is required by NVIDIA

**Answer: B** - Versioning the benchmark ensures that when you compare agent v1 vs v2, they were tested on the same cases, making the comparison valid.

---

## Key Takeaways

1. An eval pipeline has 4 parts: test dataset, agent execution, scoring, and reporting
2. Use multiple scoring methods: exact match, key fact checking, LLM-as-Judge, and RAGAS
3. NVIDIA provides AIQ Toolkit (agent eval), NeMo Evaluator (model eval), and Triton Performance Analyzer (inference eval)
4. Build custom benchmarks for your domain with 100-200+ cases across difficulty levels
5. Integrate eval into CI/CD to catch regressions before deployment
6. Always version your benchmarks so comparisons are valid over time
7. Report by category to identify specific weak areas, not just overall accuracy

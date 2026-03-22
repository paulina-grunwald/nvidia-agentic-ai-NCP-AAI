# Analyzing Evaluation Results to Guide Targeted Optimization (Objective 3.5)

How to look at evaluation data and decide what to fix, in what order, and how.

---

## The Optimization Loop

```
Evaluate → Analyze → Prioritize → Fix → Re-evaluate → Repeat
```

The key insight: don't try to fix everything at once. Evaluation results tell you WHERE the biggest problems are. Fix the highest-impact issue first, re-evaluate, then move to the next.

---

## Analyzing Evaluation Results

### Step 1: Start with the Summary

Look at overall metrics first to understand the scale of the problem:

```
Overall:
  Accuracy: 82% (target: 90%)     ← 8% gap
  Latency P50: 2.1s (target: 2s)  ← close
  Latency P99: 7.2s (target: 5s)  ← 44% over target
  Cost: $0.005/query (budget: $0.01) ← within budget
```

Summary tells you: accuracy is the main concern, P99 latency is a secondary issue, cost is fine.

### Step 2: Break Down by Category

```
Category          Accuracy   Count   Impact
order_lookup      94%        50      Low priority (already good)
troubleshooting   78%        40      HIGH - below target, common category
billing           71%        30      HIGHEST - worst accuracy
recommendations   88%        30      Medium
edge_cases        60%        20      High - but small volume

Weighted impact = (100% - accuracy) × count
  billing:         29% × 30 = 8.7  ← fix this first
  troubleshooting: 22% × 40 = 8.8  ← fix this next (highest weighted impact)
  edge_cases:      40% × 20 = 8.0  ← high failure rate but low volume
  recommendations: 12% × 30 = 3.6
  order_lookup:     6% × 50 = 3.0
```

### Step 3: Failure Analysis

For the worst category (billing), look at the actual failures:

```
Billing failures (9 of 30):
  Case 12: Asked about refund policy → Agent searched wrong KB section
  Case 15: Asked about subscription change → Agent called wrong API
  Case 18: Asked about invoice → Agent hallucinated invoice number
  Case 21: Asked about payment method → Agent gave outdated info
  Case 24: Asked about cancellation → Agent missed cancellation fee
  Case 27: Asked about promo code → Tool returned error, agent gave up
  Case 29: Asked about charge dispute → Agent used wrong dispute template
  Case 30: Asked about billing cycle → Agent confused monthly/annual
  Case 31: Asked about tax → Agent didn't have tax tool

Pattern analysis:
  Wrong tool/API called: 3 cases (33%)
  Hallucination: 2 cases (22%)
  Incomplete information: 2 cases (22%)
  Tool error not handled: 1 case (11%)
  Missing tool: 1 case (11%)
```

Now you have actionable insights. The #1 issue in billing is wrong tool selection (33%).

### Step 4: Root Cause Analysis

For each failure pattern, dig into the WHY:

```
Pattern: Wrong tool/API called (3 cases)

Root cause investigation:
  Case 12: Agent called general_kb_search instead of billing_kb_search
    → Why? Tool descriptions too similar. Agent can't distinguish them.
    → Fix: Make tool descriptions more specific

  Case 15: Agent called update_subscription instead of change_plan
    → Why? These two tools overlap. Agent picked the wrong one.
    → Fix: Consolidate into one tool or add clearer differentiation

  Case 29: Agent used generic_template instead of dispute_template
    → Why? No clear guidance on which template for which scenario
    → Fix: Add template selection logic to system prompt
```

---

## Prioritization Framework

### Impact vs Effort Matrix

```
                    High Impact
                        │
    Quick Win ──────────┼────────── Strategic Fix
    (do first)          │           (plan for next sprint)
                        │
    ─────────── Low ────┼──── High Effort ──────────
                        │
    Ignore ─────────────┼────────── Don't Bother
    (not worth it)      │           (high effort, low impact)
                        │
                    Low Impact
```

Apply this to findings:

| Finding | Impact | Effort | Priority |
|---|---|---|---|
| Tool descriptions too similar | High (33% of billing failures) | Low (prompt change) | **Quick win - do first** |
| Missing tax tool | Medium (11% of billing) | Medium (build new tool) | Plan for next sprint |
| Hallucinated invoice numbers | High (22% of billing) | Medium (add validation) | Do second |
| P99 latency too high | Medium (user experience) | High (architecture change) | Plan strategically |

### The 80/20 Rule for Agent Optimization

Usually, 20% of the issues cause 80% of the failures. Find those 20%:

```
All failure reasons across 200 test cases (36 failures):

1. Wrong tool selected:     12 cases (33%) ← Fix this
2. Hallucinated facts:       8 cases (22%) ← Fix this
3. Incomplete answer:        6 cases (17%) ← Fix this
                             ─────────
                             72% of all failures from 3 root causes

4. Tool returned error:      4 cases (11%)
5. Exceeded max iterations:  3 cases (8%)
6. Misunderstood intent:     2 cases (6%)
7. Other:                    1 case  (3%)
```

Fix items 1-3 and you address 72% of all failures.

---

## Optimization Strategies by Root Cause

### Root Cause: Wrong Tool Selection

**Indicators**: Tool accuracy < 90%, agent calls similar tools interchangeably

**Fixes (in order of effort)**:
1. Improve tool descriptions (low effort, often fixes 50%+ of cases)
2. Add few-shot examples of correct tool usage to system prompt
3. Add a tool selection pre-step: classify query first, then select tools
4. Fine-tune the model on tool selection data

### Root Cause: Hallucination

**Indicators**: Faithfulness < 0.8, claims not supported by context

**Fixes**:
1. Add "only answer based on provided context" to system prompt
2. Add self-reflection step (agent checks its own answer against context)
3. Implement CRAG (Corrective RAG) pattern to verify retrieval quality
4. Lower temperature to reduce creative generation
5. Add NeMo Guardrails fact-checking rail

### Root Cause: Incomplete Answers

**Indicators**: Answer relevance < 0.8, user feedback "incomplete"

**Fixes**:
1. Increase RAG top-K to retrieve more context
2. Increase chunk size to include more surrounding text
3. Add prompt instruction: "Provide a complete answer covering all aspects"
4. Add a completeness check step

### Root Cause: High Latency (P99)

**Indicators**: P99 > target, usually on complex queries

**Fixes (see existing eval file for details)**:
1. Profile to find bottleneck (LLM? tool calls? retrieval?)
2. If LLM: try quantization, smaller model for simple queries
3. If tools: parallelize independent tool calls, add caching
4. If retrieval: optimize vector DB, reduce chunk count
5. Set max iterations to prevent runaway loops

### Root Cause: Tool Errors

**Indicators**: Tool calls returning errors, agent not recovering

**Fixes**:
1. Improve tool error messages so agent can reason about failures
2. Add retry logic for transient errors (covered in Domain 01 error-handling.md)
3. Add fallback tools
4. Train agent to inform user gracefully when tools fail

---

## Tracking Optimization Progress

### Before/After Comparison

Every optimization should be measured:

```
Optimization: Improved billing tool descriptions

Before:
  Billing accuracy: 71%
  Tool accuracy (billing): 67%

After:
  Billing accuracy: 83% (+12%) ✅
  Tool accuracy (billing): 89% (+22%) ✅

Side effects:
  Other categories: no significant change ✅
  Latency: no change ✅
  Cost: no change ✅

Conclusion: Successful. Commit changes.
```

### Optimization Log

Keep a record of what you tried, what worked, and what didn't:

```
Date       Change                    Result           Status
03/15      Improved billing tools    +12% billing     ✅ Kept
03/17      Added self-reflection     +3% overall      ❌ Rejected (doubled latency)
03/19      Increased RAG top-K 3→5   +4% overall      ✅ Kept
03/20      Lowered temperature 0.3→0 +1% tool acc     ✅ Kept
03/22      Added reranking           +3% overall      ✅ Kept
```

This log is invaluable for understanding what works for your specific system.

---

## Continuous Optimization in Production

### Monitoring for Degradation

Once optimized, keep watching:

```python
# Alert if metrics drop below thresholds
alerts = {
    "accuracy_drop": lambda m: m.accuracy < 0.85,       # below target
    "latency_spike": lambda m: m.p99_latency > 5000,    # ms
    "error_rate_up": lambda m: m.error_rate > 0.05,      # 5%
    "cost_increase": lambda m: m.cost_per_query > 0.01,  # budget
}
```

### Data Drift Detection

Agent performance degrades when the distribution of incoming queries changes:

```
Training/eval data: mostly order lookups and troubleshooting
Production shift: suddenly 40% of queries are about a new product launch

→ Agent struggles because it wasn't tested on this type of query
→ Solution: Add new product queries to eval dataset, re-evaluate, re-optimize
```

### Re-evaluation Cadence

| Trigger | Action |
|---|---|
| New model version | Full benchmark run |
| Prompt changes | Category-level benchmark |
| New tools added | Tool accuracy benchmark |
| User feedback spike | Targeted analysis of affected category |
| Weekly | Monitor dashboard review |
| Monthly | Full benchmark + trend analysis |

---

## Exam-Style Questions

**Q1: Evaluation shows billing queries have 71% accuracy while order lookup has 94%. Where should optimization effort focus first?**
- A) Order lookup (already high, push it higher)
- B) Billing (lowest accuracy, biggest room for improvement)
- C) All categories equally
- D) Whatever is cheapest to fix

**Answer: B** - Focus on the weakest area with the biggest gap from target. Billing at 71% has the most room for improvement and likely the highest impact.

**Q2: Failure analysis shows 33% of billing errors are caused by wrong tool selection. What is the lowest-effort fix to try first?**
- A) Fine-tune the model
- B) Improve tool descriptions to make them more specific and distinct
- C) Remove similar tools
- D) Add more tools

**Answer: B** - Improving tool descriptions is a prompt-level change (low effort) that directly addresses tool confusion. Fine-tuning is high effort.

**Q3: After an optimization, billing accuracy improved from 71% to 83% but troubleshooting accuracy dropped from 78% to 75%. What should you do?**
- A) Accept the trade-off, billing was more important
- B) Investigate the troubleshooting regression since it might indicate an unintended side effect
- C) Revert all changes
- D) Ignore the 3% drop

**Answer: B** - Always investigate regressions. A 3% drop might seem small but could indicate the optimization had an unintended side effect. Check if the same change that helped billing hurt troubleshooting.

**Q4: What is the recommended cadence for full benchmark evaluation?**
- A) Every code change
- B) Daily
- C) When triggered (new model, prompt changes, feedback spikes) plus monthly as baseline
- D) Once, during initial development

**Answer: C** - Triggered evaluation catches regressions from changes. Monthly baseline catches gradual drift. Running on every tiny change is wasteful; running only once misses degradation.

---

## Key Takeaways

1. Analyze results from top-down: overall → by category → by failure case → root cause
2. Weighted impact (failure rate x volume) prioritizes what to fix first
3. The 80/20 rule: usually 2-3 root causes explain most failures
4. Use the Impact vs Effort matrix: quick wins first, then strategic fixes
5. Common root causes: wrong tool selection, hallucination, incomplete answers, high latency, tool errors
6. Always measure before/after for every optimization and check for regressions
7. Keep an optimization log so you know what worked and what didn't
8. Production requires continuous monitoring: data drift and metric degradation are ongoing risks

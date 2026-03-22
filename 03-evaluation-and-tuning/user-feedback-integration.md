# Collecting and Integrating Structured User Feedback (Objective 3.3)

How to systematically collect user feedback and feed it into the evaluation and improvement pipeline.

---

## Why Structured Feedback?

Automatic metrics (RAGAS, LLM-as-Judge) are useful but they miss what real users care about. A response might score high on faithfulness and relevance but still frustrate the user because it was too long, used jargon, or missed the real intent.

User feedback closes this gap. But unstructured feedback ("it was bad") isn't actionable. Structured feedback tells you exactly what went wrong and where.

---

## Feedback Collection Methods

### 1. Per-Response Rating

Collect feedback on individual agent responses:

```
Agent: "Your order #4521 shipped yesterday via FedEx. Expected delivery: Friday."

Rate this response:
  [👍 Helpful]  [👎 Not helpful]

If not helpful, what was wrong?
  [Incorrect] [Incomplete] [Confusing] [Too slow] [Other: ___]
```

**Schema:**
```json
{
    "response_id": "resp_abc123",
    "session_id": "sess_xyz789",
    "rating": "negative",
    "issue_type": "incomplete",
    "user_comment": "Didn't include tracking number",
    "timestamp": "2025-03-22T10:30:00Z",
    "agent_version": "v2.3",
    "query_category": "order_lookup"
}
```

### 2. Session-Level Feedback

After the full interaction:

```
How was your overall experience?
  ⭐⭐⭐⭐☆ (4/5)

Was your issue resolved?
  [Yes, completely] [Partially] [No]

Would you use this agent again?
  [Yes] [Maybe] [No]
```

### 3. Correction Feedback

The most valuable type: user provides the correct answer.

```
Agent: "Your subscription renews on April 1st."
User: "That's wrong, it renews on March 15th."

Captured correction:
{
    "type": "factual_correction",
    "agent_claim": "subscription renews on April 1st",
    "user_correction": "renews on March 15th",
    "source": "user_direct_correction"
}
```

This directly creates training/evaluation data: you now have an input, a wrong output, and the correct output.

### 4. Implicit Feedback

Behavioral signals that don't require user action:

| Signal | Interpretation | Collection Method |
|---|---|---|
| **Regeneration** | User didn't like the response | Track "retry" clicks |
| **Query reformulation** | Agent misunderstood | Compare consecutive queries for semantic similarity |
| **Session abandonment** | User gave up | Track sessions without resolution |
| **Time to next message** | Short = good flow, Long = confusion | Timestamp analysis |
| **Copy/paste** | Response was useful | Track clipboard events |
| **Escalation to human** | Agent couldn't handle it | Track handoff events |

---

## Feedback Data Pipeline

```
Collection          Processing           Storage           Usage

User ratings  →  Validate &     →  Feedback DB   →  Evaluation dataset
Corrections      Categorize        (structured)     Prompt optimization
Implicit      →  Aggregate      →  Analytics DB  →  Model fine-tuning
signals          by pattern         (metrics)       Quality dashboards
```

### Processing Steps

1. **Validate**: Remove spam, bots, duplicate submissions
2. **Categorize**: Tag by issue type (accuracy, completeness, tone, latency)
3. **Link to traces**: Connect feedback to the full agent trace (reasoning steps, tool calls)
4. **Aggregate**: Calculate metrics per category, per time period, per agent version

### Linking Feedback to Agent Traces

This is critical. A thumbs-down alone isn't useful. But a thumbs-down linked to the full trace tells you:
- What the user asked
- What the agent reasoned
- Which tools it called
- What it answered
- Why the user was unhappy

```python
# When collecting feedback, include the trace ID
feedback_record = {
    "trace_id": "trace_abc123",  # links to full agent execution trace
    "rating": "negative",
    "issue": "wrong_answer",
    "user_correction": "The correct policy is..."
}

# Later, analyze: pull trace + feedback together
trace = get_trace("trace_abc123")
# Now you can see: the agent called the wrong tool → got wrong data → gave wrong answer
# Root cause: tool description was ambiguous
```

---

## From Feedback to Improvement

### Building Evaluation Datasets from Feedback

Turn user corrections into test cases:

```python
# User corrected the agent → new test case
new_test_case = {
    "input": feedback.original_query,
    "expected_output": feedback.user_correction,
    "expected_tools": trace.tools_used,  # or corrected tools
    "source": "user_feedback",
    "date_added": "2025-03-22"
}

evaluation_dataset.append(new_test_case)
```

Over time, your eval dataset grows from real user interactions, making it more representative than synthetic test cases.

### Feedback-Driven Prompt Optimization

Aggregate feedback patterns to identify prompt issues:

```
Analysis: Last 30 days, 500 feedback records

Top issues:
1. "Incomplete answer" - 35% of negative feedback
   → Most common in "policy_lookup" category
   → Agent retrieves correct doc but only quotes part of it
   → Fix: Add to prompt "Always provide the complete policy text"

2. "Too verbose" - 22% of negative feedback
   → Most common in "simple_question" category
   → Agent adds unnecessary caveats and explanations
   → Fix: Add "Be concise for simple questions"

3. "Wrong tool used" - 18% of negative feedback
   → Agent uses web search when internal KB has the answer
   → Fix: Refine tool descriptions (covered in Domain 02)
```

### RLHF and Preference Learning

For larger-scale improvement, user feedback can train reward models:

```
Feedback pairs:
  Query: "What's NVIDIA's stock price?"
  Response A: "NVIDIA (NVDA) is currently trading at $875.50" [👍]
  Response B: "I don't have real-time data" [👎]
  → Preference: A > B

These pairs train a reward model that scores responses.
The reward model then guides:
  - Prompt optimization (which prompt gets higher reward scores)
  - Fine-tuning (RLHF/DPO to align model with user preferences)
```

This connects to the NVIDIA NeMo RLHF training pipeline, where user preference data is used to improve model alignment.

---

## Feedback Metrics Dashboard

### Key Metrics to Track

| Metric | Formula | Target |
|---|---|---|
| **User satisfaction score** | Avg rating across sessions | > 4.0 / 5.0 |
| **Resolution rate** | Sessions marked "resolved" / total | > 80% |
| **Negative feedback rate** | Thumbs-down / total responses | < 10% |
| **Correction frequency** | User corrections / total responses | < 5% |
| **Escalation rate** | Handoffs to human / total sessions | < 15% |
| **Feedback response rate** | Users who give feedback / total users | > 20% (good engagement) |

### Trend Analysis

```
Week    Satisfaction   Neg Rate   Corrections   Escalations
W1      3.8            15%        8%            22%
W2      4.0            12%        6%            18%        ← prompt fix deployed
W3      4.1            10%        5%            16%
W4      4.2            9%         4%            14%
W5      3.9            14%        7%            19%        ← regression! investigate
```

Week 5 regression: check what changed (new model? prompt change? new data?).

---

## Exam-Style Questions

**Q1: Which type of user feedback is most directly useful for creating new evaluation test cases?**
- A) Star ratings (1-5)
- B) User corrections with the right answer
- C) Thumbs up/down
- D) Session duration

**Answer: B** - Corrections provide both the wrong output and the correct output, directly creating an input/expected-output pair for the evaluation dataset.

**Q2: An agent has a 92% accuracy on automatic benchmarks but users give it a 3.2/5.0 satisfaction score. What does this suggest?**
- A) The benchmarks are wrong
- B) Users are too harsh
- C) The benchmarks don't capture what users care about (completeness, tone, speed, etc.)
- D) The agent is fine, ignore user feedback

**Answer: C** - Automatic metrics and user satisfaction can diverge. Users care about things benchmarks might not measure: verbosity, tone, response time, completeness of explanation.

**Q3: To understand why a user gave a thumbs-down, what additional data is most useful?**
- A) The user's account information
- B) The full agent execution trace linked to the feedback
- C) Other users' feedback on different queries
- D) The model's parameter count

**Answer: B** - Linking feedback to the full trace (query, reasoning, tools, answer) lets you see exactly where things went wrong, enabling targeted fixes.

**Q4: Feedback analysis shows 35% of negative ratings cite "incomplete answers" in the policy lookup category. What is the most targeted fix?**
- A) Switch to a larger model
- B) Add more tools
- C) Refine the system prompt to instruct the agent to provide complete policy text
- D) Remove the policy lookup feature

**Answer: C** - The feedback points to a specific issue (incomplete) in a specific category (policy lookup). A targeted prompt fix addresses this directly without changing the entire system.

---

## Key Takeaways

1. Structured feedback (typed issues, corrections) is far more actionable than simple ratings
2. Correction feedback creates direct evaluation data: wrong output + correct output pairs
3. Always link feedback to the full agent trace so you can diagnose root causes
4. Implicit feedback (regenerations, abandonment, reformulation) doesn't require user effort
5. Aggregate feedback patterns to identify systemic issues (not just individual failures)
6. Feedback-driven eval datasets grow more representative over time vs static benchmarks
7. Track feedback metrics as trends. Sudden changes signal regressions or improvements.
8. RLHF/DPO uses preference pairs from user feedback to align model behavior

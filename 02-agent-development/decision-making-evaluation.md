# Evaluating and Refining Agent Decision-Making

How to systematically assess and improve the decisions agents make.

---

## What Is Agent Decision-Making?

Every turn, the agent makes decisions:

- Should I use a tool or respond directly?
- Which tool should I call?
- What arguments should I pass?
- Is this answer good enough, or should I refine it?
- Should I ask for clarification or take my best guess?

These decisions compound. A bad tool selection early in a ReAct loop can waste multiple steps. Evaluating decision quality means looking at the whole chain, not just the final output.

---

## Decision Quality Metrics

### Tool Selection Accuracy

Did the agent pick the right tool for the situation?

```
Scenario: User asks "What's NVIDIA's stock price today?"

Good decision: Call get_stock_price(ticker="NVDA")
Bad decision:  Call search_web("NVIDIA stock price") → less reliable
Worse:         Try to answer from memory → likely outdated

Metric: % of turns where the agent selected the optimal tool
```

### Action Efficiency

How many steps did the agent take vs. the minimum needed?

```
Optimal path: 3 steps (classify → search → respond)
Agent's path: 7 steps (search wrong DB → retry → search another → fail → search right DB → verify → respond)

Efficiency = optimal_steps / actual_steps = 3/7 = 43%
```

Target varies by use case, but generally you want >70% efficiency.

### Reasoning Quality

Is the agent's chain of thought logical and correct?

Evaluate intermediate reasoning steps, not just the final answer:

```
Step 1: "I need to find the customer's order" ✅ Correct intent
Step 2: "Let me search by email" ✅ Reasonable approach
Step 3: "No results. Let me try order number" ✅ Good recovery
Step 4: "Found it. Status: shipped" ✅ Correct result

Score: 4/4 reasoning steps correct
```

### Final Answer Quality

Standard metrics for the output:

- **Accuracy**: Is the answer factually correct?
- **Completeness**: Did it address all parts of the question?
- **Groundedness**: Is the answer supported by retrieved/tool data (not hallucinated)?
- **Relevance**: Does the answer actually address what was asked?

### Hallucination Rate

How often does the agent make claims not supported by its tools/data?

```
Tool returned: "Order shipped March 15, arriving March 20"
Agent said: "Your order shipped March 15 and will arrive March 20"  ✅ grounded
Agent said: "Your order shipped March 15 and will arrive March 18"  ❌ hallucinated arrival date
```

---

## Evaluation Methods

### 1. Trajectory Evaluation

Evaluate the entire sequence of actions, not just the final answer. This is the most important evaluation method for agents.

```
Trajectory:
  1. Thought: "I need customer data" → ✅ correct
  2. Action: search_customers(name="John Smith") → ✅ right tool, right args
  3. Observation: {customer_id: 123, orders: [...]} → (external data)
  4. Thought: "Found the customer, now check recent orders" → ✅ logical
  5. Action: get_orders(customer_id=123, limit=5) → ✅ right tool
  6. Thought: "The most recent order is #456, shipped yesterday" → ✅ correct interpretation
  7. Final answer: "Your most recent order #456 shipped yesterday" → ✅ accurate

Trajectory score: 6/6 decision points correct
```

### 2. A/B Testing for Agent Strategies

Compare two different agent configurations on the same inputs:

```
Strategy A: ReAct with self-reflection (more thorough, slower)
Strategy B: Simple ReAct without reflection (faster, less thorough)

Test set: 200 customer support queries

Results:
  Strategy A: 89% accuracy, 4.2s avg latency, 5.1 avg steps
  Strategy B: 82% accuracy, 2.1s avg latency, 3.2 avg steps

Decision: Use Strategy A for complex queries, B for simple ones
```

### 3. Comparative Evaluation

Have two agent versions handle the same queries, then judge which is better:

```python
# Run same query through both agents
result_v1 = agent_v1.run("Explain NVIDIA NIM")
result_v2 = agent_v2.run("Explain NVIDIA NIM")

# Use LLM-as-judge to compare
judge_prompt = f"""
Compare these two responses to the query "Explain NVIDIA NIM":

Response A: {result_v1}
Response B: {result_v2}

Which response is better and why? Consider accuracy, completeness, and clarity.
"""
judgment = judge_llm.invoke(judge_prompt)
```

### 4. LLM-as-Judge

Use a strong LLM to evaluate agent outputs:

```python
evaluation_prompt = """
Rate this agent response on a scale of 1-5 for each dimension:

Query: {query}
Agent Response: {response}
Ground Truth: {ground_truth}  # optional

Dimensions:
- Accuracy (1-5): Is the response factually correct?
- Completeness (1-5): Does it fully address the query?
- Groundedness (1-5): Is the response based on retrieved data, not hallucinated?
- Helpfulness (1-5): Would the user find this useful?

Provide scores and brief justification for each.
"""
```

This is scalable (no human annotators needed) and correlates well with human judgment for most tasks.

### 5. Human Evaluation

For high-stakes domains, human evaluation is still the gold standard:

| Method                            | Speed     | Cost      | Quality                          |
| --------------------------------- | --------- | --------- | -------------------------------- |
| LLM-as-Judge                      | Fast      | Low       | Good for most cases              |
| Single human rater                | Slow      | Medium    | Good                             |
| Multiple human raters + agreement | Very slow | High      | Best (inter-rater reliability)   |
| Domain expert review              | Slow      | Very high | Required for specialized domains |

---

## Refinement Strategies

Once you've identified where decisions are poor, here's how to improve them.

### 1. Prompt Refinement

Most decision-making issues can be fixed by improving prompts:

```
Problem: Agent uses web search for questions answerable from the knowledge base
Fix: Add to system prompt:
     "ALWAYS check the internal knowledge base first before using web search.
      Only use web search if the knowledge base returns no relevant results."

Problem: Agent gives overly long answers
Fix: Add to system prompt:
     "Keep responses concise. Aim for 2-3 sentences unless the user asks for detail."
```

### 2. Tool Description Refinement

If the agent misuses tools, improve the descriptions:

```
Before: description="Search for information"
Problem: Agent uses this for everything

After: description="Search the internal product catalog for product names, prices,
       and availability. Do NOT use for general knowledge questions."
Problem fixed: Agent only uses it for product queries
```

### 3. Few-Shot Decision Examples

Add examples of correct decisions to the prompt:

```
Here are examples of correct tool usage:

User: "What's the weather?" → Use: get_weather(city)
User: "Calculate 15% tip on $80" → Use: calculator(expression)
User: "Tell me about machine learning" → Use: NO TOOL (answer from knowledge)
User: "What's my order status?" → Use: lookup_order(customer_id)
```

### 4. Self-Reflection Loop

Add a verification step where the agent checks its own reasoning:

```python
def agent_with_reflection(query):
    # Initial response
    response = agent.run(query)

    # Self-check
    check = llm.invoke(f"""
    Review this agent response for errors:
    Query: {query}
    Response: {response}

    Check for:
    1. Factual accuracy (any hallucinated claims?)
    2. Tool usage (was the right tool used?)
    3. Completeness (all parts of the query addressed?)

    If issues found, provide a corrected response.
    If no issues, respond with "APPROVED".
    """)

    if "APPROVED" not in check:
        return check  # corrected response
    return response
```

### 5. Strategy Routing

Use different strategies for different query types:

```python
def route_strategy(query):
    complexity = classify_complexity(query)

    if complexity == "simple":
        # Direct answer, no tools, no reflection
        return simple_agent.run(query)
    elif complexity == "moderate":
        # ReAct with tools
        return react_agent.run(query)
    elif complexity == "complex":
        # ReAct + self-reflection + multiple attempts
        return thorough_agent.run(query)
```

This optimizes latency for simple queries while maintaining accuracy for complex ones.

---

## Iterative Refinement Workflow

The full improvement cycle:

```
1. Deploy agent v1
   ↓
2. Collect interaction logs and feedback
   ↓
3. Run evaluation pipeline (trajectory eval, LLM-as-judge)
   ↓
4. Identify failure patterns:
   - "Agent hallucinates product prices 15% of the time"
   - "Agent uses web search when knowledge base would be faster"
   - "Agent asks too many clarifying questions for simple queries"
   ↓
5. Apply fixes:
   - Add tool description: "Always check product DB for prices"
   - Add few-shot examples of when NOT to search the web
   - Add system prompt: "For simple queries, answer directly"
   ↓
6. A/B test v1 vs v2 on held-out test set
   ↓
7. If v2 is better → deploy
   ↓
8. Return to step 2
```

This connects to Domain 03 (Evaluation and Tuning). The evaluation pipeline from Domain 03 feeds back into decision-making refinement here.

---

## Exam-Style Questions

**Q1: An agent frequently calls the web search tool for questions that could be answered from its internal knowledge base. What is the best fix?**

- A) Remove the web search tool entirely
- B) Refine the tool descriptions and system prompt to prioritize the knowledge base
- C) Increase the LLM temperature
- D) Add more tools

**Answer: B** - Improve the system prompt to specify "check knowledge base first" and refine the web search description to specify when it should (and shouldn't) be used.

**Q2: Which evaluation method is most appropriate for assessing the quality of an agent's step-by-step reasoning, not just its final answer?**

- A) Final answer accuracy
- B) Trajectory evaluation
- C) Latency measurement
- D) Token count

**Answer: B** - Trajectory evaluation assesses each decision point in the agent's reasoning chain, identifying where errors occur even if the final answer happens to be correct.

**Q3: An agent is being compared in two configurations: Strategy A (thorough, slow) and Strategy B (fast, less thorough). What evaluation approach directly compares them?**

- A) Unit testing
- B) A/B testing on the same test set
- C) Code review
- D) Load testing

**Answer: B** - A/B testing runs both strategies on the same inputs and compares accuracy, latency, and efficiency to determine which is better (or when to use each).

**Q4: What is the primary advantage of using LLM-as-Judge for agent evaluation?**

- A) It's always more accurate than human evaluation
- B) It's scalable and cost-effective compared to human evaluation
- C) It doesn't require any prompt engineering
- D) It only evaluates final answers

**Answer: B** - LLM-as-Judge scales to thousands of evaluations at low cost, while human evaluation is expensive and slow. It's "good enough" for most cases, though human eval remains gold standard for specialized domains.

---

## Key Takeaways for NVIDIA Certification

1. Agent decisions include tool selection, argument generation, reasoning steps, and answer formulation
2. Trajectory evaluation assesses the entire decision chain, not just the final output
3. Key metrics: tool selection accuracy, action efficiency, reasoning quality, hallucination rate
4. A/B testing compares agent strategies on the same inputs to pick the best approach
5. LLM-as-Judge is the scalable alternative to human evaluation
6. Most decision-making issues are fixed through prompt and tool description refinement
7. Self-reflection loops add a verification step that catches errors before responding
8. Strategy routing sends simple queries to fast agents and complex queries to thorough agents
9. The iterative refinement cycle: deploy → collect feedback → evaluate → fix → A/B test → redeploy


# Reasoning Patterns

Reasoning patterns define **how an agent thinks, plans, and decides** before or during action. These patterns determine the agent's problem-solving strategy and are central to deliberative architectures.

## 1. Chain-of-Thought (CoT)

CoT is a prompting technique where the LLM **explains its reasoning step-by-step** before giving a final answer. All reasoning happens **inside the model's text output** - no tools, no external data.

```
Input → Think step 1 → Think step 2 → Think step 3 → Answer
```

- The model breaks a complex problem into smaller logical steps
- Each step builds on the previous one
- Reasoning is fully transparent in the output text
- No interaction with the outside world

| Aspect | CoT |
| --- | --- |
| Actions/Tools | No |
| External data | No (uses only training knowledge) |
| Reasoning visible | Yes (in output text) |
| Best for | Math, logic, analysis tasks |

**Example:**
```
Q: If a store has 3 shelves with 8 books each, and 5 books are sold, how many remain?

Step 1: Total books = 3 shelves x 8 books = 24 books
Step 2: Books sold = 5
Step 3: Remaining = 24 - 5 = 19

Answer: 19 books remain.
```

## 2. ReAct (Reasoning + Acting)

ReAct **interleaves reasoning with actions**. The agent thinks, then acts (calls tools/APIs), observes results, then thinks again. This is the dominant pattern for agentic AI systems.

```
Thought → Action → Observation → Thought → Action → Observation → ... → Answer
```

- Combines internal reasoning with external tool use
- Adapts its plan based on real-world observations
- Can access real-time data, databases, APIs
- Self-correcting - adjusts based on what it learns

| Aspect | ReAct |
| --- | --- |
| Actions/Tools | Yes (APIs, databases, search, etc.) |
| External data | Yes (retrieves real-time info) |
| Reasoning visible | Yes (Thought steps) |
| Best for | Tasks requiring tools, real-time data, multi-step actions |

**Example:**
```
Thought 1: I need to find the weather in San Francisco.
Action 1: search_weather(location="San Francisco")
Observation 1: { "temperature": "18C", "precipitation_chance": "75%" }

Thought 2: 75% rain chance - the user should bring an umbrella.
Action 2: final_answer()

Answer: It's 18C and cloudy with 75% chance of rain. Bring an umbrella.
```

## 3. Tree of Thoughts (ToT)

ToT explores **multiple reasoning paths simultaneously**, like a decision tree. Instead of following one chain, it branches out, evaluates each path, and selects the best one. Think of it as CoT but exploring several options in parallel.

```
         ┌─ Path A → evaluate → score
Input ───┼─ Path B → evaluate → score  → pick best → Answer
         └─ Path C → evaluate → score
```

- Generates multiple candidate reasoning chains
- Evaluates and scores each path (using heuristics or LLM self-evaluation)
- Can backtrack from dead ends and explore alternatives
- Uses search strategies like BFS (breadth-first) or DFS (depth-first)

| Aspect | ToT |
| --- | --- |
| Actions/Tools | Optional |
| External data | Optional |
| Reasoning visible | Yes (multiple branches) |
| Best for | Creative tasks, puzzles, strategic planning, problems with multiple valid approaches |

**Example (solving a puzzle):**
```
Path A: Start with corner pieces → evaluate → score: 6/10
Path B: Sort by color first → evaluate → score: 8/10
Path C: Find edge pieces → evaluate → score: 7/10

Best path: B (sort by color first)
→ Continue reasoning along Path B...
```

## 4. Self-Reflection

Self-Reflection adds a **critique-and-revise loop** where the agent evaluates its own output, identifies errors or weaknesses, and iterates to improve. The agent acts as both generator and reviewer.

```
Generate answer → Critique own answer → Revise → Critique again → ... → Final answer
```

- Agent reviews its own reasoning and output for correctness
- Identifies gaps, errors, or hallucinations
- Iteratively refines until quality threshold is met
- Can incorporate external feedback (human-in-the-loop)

| Aspect | Self-Reflection |
| --- | --- |
| Actions/Tools | Optional |
| External data | Optional |
| Reasoning visible | Yes (critique + revision visible) |
| Best for | High-quality outputs, error-prone tasks, code generation, writing |

**Example:**
```
Draft: "Python was created in 1989 by Guido van Rossum."

Self-critique: The first release was actually in 1991. 1989 is when development
started but the question asks about creation/release. Let me correct this.

Revised: "Python was first released in 1991 by Guido van Rossum, with development
starting in 1989."
```

---

## Comparison Table

| Pattern | Best For | Complexity | Cost | When to Use |
| --- | --- | --- | --- | --- |
| **ReAct** | Multi-step tasks with tools | Medium | Medium | Dynamic problems needing external data |
| **CoT** | Logical/math reasoning | Low | Low | Breaking down complex reasoning |
| **ToT** | Finding optimal solutions | High | High | Creative tasks, puzzles, strategic planning |
| **Self-Reflection** | High-quality outputs | Medium | Medium-High | When quality matters, error-prone tasks |

## CoT vs ReAct: Side-by-Side

| Aspect | Chain-of-Thought | ReAct |
| --- | --- | --- |
| **Core idea** | Think step-by-step | Think AND act |
| **Tool use** | No | Yes |
| **External APIs** | No | Yes |
| **Real-time data** | No | Yes |
| **Output format** | Reasoning → Answer | Thought → Action → Observation → ... |
| **Use case** | Logic, math, analysis | Agents, assistants, RAG |
| **Complexity** | Simpler | More complex |

CoT = "Let me think through this..."
ReAct = "Let me think AND do something about it..."

## When to Use Which

| Scenario | Best Pattern | Why |
| --- | --- | --- |
| Math word problems | CoT | Pure reasoning, no external data needed |
| Logic puzzles | CoT | Pure reasoning |
| Code explanation | CoT | Analyzing existing code |
| Weather queries | ReAct | Needs API call for real-time data |
| RAG applications | ReAct | Needs to retrieve documents |
| Database lookups | ReAct | Needs to query external systems |
| Multi-step agents | ReAct | Needs to take actions |
| Booking flights | ReAct | Needs to call booking APIs |
| Crossword / puzzle solving | ToT | Multiple paths, need to explore and backtrack |
| Strategic game play | ToT | Evaluate many moves ahead |
| Essay writing | Self-Reflection | Iterative quality improvement |
| Code generation | Self-Reflection | Catch bugs through self-review |
| Factual Q&A (high stakes) | Self-Reflection | Reduce hallucinations via self-check |

## Key Takeaways for NVIDIA Certification

1. **CoT** = pure reasoning, no tools, low cost - best for logic/math
2. **ReAct** = reasoning + tool use, the dominant agentic pattern - used in LangChain, NeMo agents
3. **ToT** = branching exploration, high cost - best when you need to evaluate multiple approaches
4. **Self-Reflection** = generate-critique-revise loop - best when output quality is critical
5. Patterns can be **combined** - e.g., ReAct with Self-Reflection for tool-using agents that verify their own outputs

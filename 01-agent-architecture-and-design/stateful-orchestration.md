# Stateful Orchestration and Prompt Chains

Applying logic trees, prompt chains, and stateful orchestration for multi-step reasoning.

---

## Prompt Chains

A prompt chain is a sequence of LLM calls where the output of one call feeds into the next. Each step has a focused prompt for a specific subtask, rather than one giant prompt trying to do everything.

### Why Chain Instead of One Big Prompt?

| Approach      | Pros                                              | Cons                                         |
| ------------- | ------------------------------------------------- | -------------------------------------------- |
| Single prompt | Simple, one LLM call                              | Unreliable for complex tasks, hard to debug  |
| Prompt chain  | More reliable, easier to debug, each step focused | Multiple LLM calls (more latency, more cost) |

Rule of thumb: if a task has more than 2-3 distinct steps, break it into a chain.

### Sequential Chain

Output of step N becomes input of step N+1:

```
Step 1: "Extract key facts from this document"
   → facts
Step 2: "Categorize these facts into themes: {facts}"
   → categorized_facts
Step 3: "Write an executive summary based on these categorized facts: {categorized_facts}"
   → summary
```

Each step gets a focused prompt. The LLM only has to handle one task at a time.

### Conditional Chain (Branching)

Different paths based on the output of a step:

```
Step 1: "Classify this customer query: {query}"
   → classification

IF classification == "billing":
    Step 2a: "Look up billing info for customer: {customer_id}"
    Step 3a: "Draft billing response based on: {billing_info}"

IF classification == "technical":
    Step 2b: "Search knowledge base for: {query}"
    Step 3b: "Draft technical solution based on: {kb_results}"

IF classification == "complaint":
    Step 2c: "Analyze sentiment and severity of: {query}"
    Step 3c: "Draft empathetic response with escalation if needed"
```

This is basically a decision tree where each branch has its own chain of prompts.

### Parallel Chain (Fan-Out / Fan-In)

Multiple chains run simultaneously, results are merged:

```
              ┌→ Chain A: "Analyze market data" → market_analysis
Input query → ├→ Chain B: "Analyze competitors" → competitor_analysis  → Merge → Final report
              └→ Chain C: "Analyze financials"  → financial_analysis
```

Faster than sequential when subtasks are independent.

### Transform Chain

Each step transforms the data format without adding new information:

```
Raw text → Extract entities → Convert to table → Format as markdown → Output
```

Useful for data processing pipelines.

---

## Stateful Orchestration with LangGraph

LangGraph is the primary framework for building stateful agent workflows. It models workflows as directed graphs where nodes are processing steps and edges define the flow.

### Core Concepts

**State**: A typed dictionary that persists across all nodes. Every node can read and write to it.

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # conversation history (appends)
    current_step: str                         # tracks where we are
    classification: str                       # result of classification step
    tool_results: list                        # accumulated tool outputs
    final_answer: str                         # the answer to return
```

**Nodes**: Functions that take state as input, do some processing, and return updated state.

```python
def classify_node(state: AgentState) -> dict:
    # Use LLM to classify the user's query
    result = llm.invoke(f"Classify this query: {state['messages'][-1]}")
    return {"classification": result, "current_step": "classified"}

def research_node(state: AgentState) -> dict:
    # Do research based on the classification
    results = search_tool.run(state["messages"][-1])
    return {"tool_results": [results], "current_step": "researched"}
```

**Edges**: Define the flow between nodes. Can be fixed or conditional.

```python
# Fixed edge: always go from A to B
graph.add_edge("classify", "route")

# Conditional edge: go to different nodes based on state
def route_decision(state: AgentState) -> str:
    if state["classification"] == "simple":
        return "direct_answer"
    elif state["classification"] == "complex":
        return "research"
    else:
        return "ask_clarification"

graph.add_conditional_edges("route", route_decision, {
    "direct_answer": "answer_node",
    "research": "research_node",
    "ask_clarification": "clarify_node"
})
```

### Building a Complete Stateful Workflow

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("classify", classify_node)
graph.add_node("research", research_node)
graph.add_node("answer", answer_node)
graph.add_node("clarify", clarify_node)

# Set entry point
graph.set_entry_point("classify")

# Add conditional routing
graph.add_conditional_edges("classify", route_decision, {
    "direct_answer": "answer",
    "research": "research",
    "ask_clarification": "clarify"
})

# Research always leads to answer
graph.add_edge("research", "answer")

# Answer and clarify lead to END
graph.add_edge("answer", END)
graph.add_edge("clarify", END)

# Compile
app = graph.compile()
```

Visualized:

```
START → classify → [conditional] → research → answer → END
                                 → answer → END
                                 → clarify → END
```

---

## State Machines for Agent Workflows

A state machine is a model where the system is always in exactly one state, and transitions between states are triggered by events or conditions. This maps perfectly to agent workflows.

### Agent as a State Machine

```
States: IDLE, THINKING, ACTING, WAITING_FOR_TOOL, REFLECTING, DONE, ERROR

Transitions:
  IDLE + user_message → THINKING
  THINKING + need_tool → ACTING
  ACTING + tool_call → WAITING_FOR_TOOL
  WAITING_FOR_TOOL + tool_result → THINKING (loop back)
  THINKING + have_answer → REFLECTING
  REFLECTING + quality_ok → DONE
  REFLECTING + quality_bad → THINKING (try again)
  ANY_STATE + error → ERROR
  ERROR + retry → THINKING
  ERROR + max_retries → DONE (with error message)
```

This is exactly what LangGraph implements. Each node is a state, conditional edges are transitions.

### Why State Machines Matter

1. **Predictability**: You know exactly what states the agent can be in
2. **Debugging**: When something goes wrong, you can inspect which state the agent was in
3. **Checkpointing**: Save state at any point, resume from there
4. **Human-in-the-loop**: Insert approval gates as states
5. **Testing**: Test each state and transition independently

---

## Conditional Routing and Branching

### Simple Conditional

Based on a single condition:

```python
def should_continue(state):
    if state["error_count"] > 3:
        return "give_up"
    if state["has_answer"]:
        return "finalize"
    return "continue_reasoning"
```

### Multi-Factor Routing

Based on multiple conditions:

```python
def complex_router(state):
    classification = state["classification"]
    confidence = state["confidence"]
    has_tools = state["available_tools"]

    if classification == "simple" and confidence > 0.9:
        return "direct_answer"
    elif classification == "complex" and has_tools:
        return "tool_assisted"
    elif confidence < 0.5:
        return "ask_clarification"
    else:
        return "research_first"
```

### Loop with Exit Condition

Agents often loop (think → act → observe → think...) until they have an answer or hit a limit:

```python
def should_loop(state):
    # Exit conditions
    if state["iterations"] >= 10:       # safety limit
        return "force_finish"
    if state["has_final_answer"]:
        return "finish"
    return "continue"                    # loop back

graph.add_conditional_edges("reasoning", should_loop, {
    "continue": "act",
    "finish": "format_answer",
    "force_finish": "format_answer"
})
graph.add_edge("act", "observe")
graph.add_edge("observe", "reasoning")   # loop
```

---

## Logic Trees for Decision-Making

A logic tree is a hierarchical breakdown of a decision into sub-decisions. Agents use them to structure complex reasoning.

### Example: Loan Approval Agent

```
Should we approve the loan?
├── Is the applicant eligible?
│   ├── Age >= 18? [check]
│   ├── Valid ID? [check]
│   └── Not on sanctions list? [tool call]
├── Is the financial risk acceptable?
│   ├── Credit score > 650? [tool call]
│   ├── Debt-to-income ratio < 40%? [calculation]
│   └── Employment verified? [tool call]
├── Does the loan meet policy?
│   ├── Amount within limits? [check]
│   ├── Term within limits? [check]
│   └── Rate appropriate for risk? [calculation]
└── Final decision
    ├── All checks passed → APPROVE
    ├── Financial risk failed → DENY (with reason)
    └── Missing information → REQUEST MORE INFO
```

Each leaf node is either a check, a calculation, or a tool call. The tree structure ensures nothing is missed.

### Implementing Logic Trees as Prompt Chains

```python
# Step 1: Eligibility check
eligibility = llm.invoke("Check eligibility criteria for: {applicant_data}")

# Step 2: Financial risk (only if eligible)
if eligibility.passed:
    risk = llm.invoke("Assess financial risk for: {financial_data}")
else:
    return "DENIED: Eligibility failed"

# Step 3: Policy check (only if risk is acceptable)
if risk.acceptable:
    policy = llm.invoke("Check policy compliance for: {loan_details}")
else:
    return "DENIED: Financial risk too high"

# Step 4: Final decision
if policy.compliant:
    return "APPROVED"
else:
    return f"DENIED: {policy.reason}"
```

---

## Multi-Step Reasoning Orchestration

### DAG Workflows (Directed Acyclic Graph)

Most real agent workflows aren't simple chains. They're DAGs where some steps can run in parallel and some depend on others.

```
                ┌→ B (market research)  ─┐
A (plan) ──────→├→ C (data analysis)    ─├──→ E (synthesize) → F (output)
                └→ D (competitor scan)  ─┘
```

B, C, D run in parallel after A. E waits for all three. F runs after E.

### Cycles (Loops)

Unlike DAGs, some workflows need cycles. ReAct is inherently cyclic:

```
Think → Act → Observe → Think → Act → Observe → ... → Final Answer
```

LangGraph supports cycles natively. You define a loop and an exit condition:

```python
graph.add_edge("act", "observe")
graph.add_edge("observe", "think")
graph.add_conditional_edges("think", check_done, {
    "continue": "act",    # cycle
    "done": "finalize"    # exit
})
```

### Conditional Loops with Backtracking

More advanced: the agent can go back to an earlier step if the current approach isn't working.

```
Plan → Execute Step 1 → Evaluate → [OK] → Execute Step 2 → ...
                                 → [FAIL] → Replan (go back to Plan)
```

This is related to Tree of Thoughts: if one branch fails, backtrack and try another.

---

## Checkpointing and Persistence

Stateful orchestration isn't useful without persistence. If the system crashes mid-workflow, you need to resume from where you left off.

### LangGraph Checkpointing

```python
from langgraph.checkpoint.postgres import PostgresSaver

# Set up persistent checkpointing
checkpointer = PostgresSaver.from_conn_string("postgresql://...")

# Compile graph with checkpointer
app = graph.compile(checkpointer=checkpointer)

# Each invocation uses a thread_id for continuity
config = {"configurable": {"thread_id": "user-session-123"}}
result = app.invoke(initial_state, config=config)

# Later: resume from last checkpoint
state = app.get_state(config)
# Modify state if needed (human-in-the-loop)
app.update_state(config, {"approved": True})
# Continue from modified state
result = app.invoke(None, config=config)
```

Checkpointing enables:

- **Fault tolerance**: Resume after crashes
- **Human-in-the-loop**: Pause, let human review/modify, continue
- **Audit trail**: Full history of all state changes
- **Time travel**: Go back to any previous state

### What Gets Checkpointed

| Component              | Saved? | Why                  |
| ---------------------- | ------ | -------------------- |
| Conversation messages  | Yes    | Maintain context     |
| Agent decisions        | Yes    | Audit trail          |
| Tool call results      | Yes    | Avoid re-executing   |
| Intermediate reasoning | Yes    | Debug and review     |
| Current node/step      | Yes    | Know where to resume |

---

## Workflow Patterns Summary

| Pattern            | Structure              | Use Case                    | LangGraph Support            |
| ------------------ | ---------------------- | --------------------------- | ---------------------------- |
| Sequential chain   | A → B → C              | Simple pipelines            | add_edge                     |
| Conditional branch | A → [if X: B, else: C] | Classification + routing    | add_conditional_edges        |
| Parallel fan-out   | A → [B, C, D] → E      | Independent subtasks        | Multiple edges from one node |
| Loop (cycle)       | A → B → A (repeat)     | ReAct, iterative refinement | Cycles with exit conditions  |
| DAG                | Complex dependencies   | Real-world workflows        | Full graph support           |
| Hierarchical       | Supervisor → Workers   | Multi-agent coordination    | Nested graphs/subgraphs      |

---

## Exam-Style Questions

**Q1: An agent needs to classify a customer query and then route it to different processing pipelines based on the category. Which LangGraph feature enables this?**

- A) add_edge
- B) add_conditional_edges
- C) add_node
- D) set_entry_point

**Answer: B** -`add_conditional_edges` allows routing to different nodes based on a function that inspects the current state.

**Q2: What is the primary advantage of prompt chains over a single large prompt?**

- A) Lower cost (fewer tokens)
- B) Faster execution (parallel processing)
- C) More reliable results (each step is focused on one task)
- D) Simpler implementation

**Answer: C** -Prompt chains break complex tasks into focused steps, making each step more reliable. The trade-off is more LLM calls.

**Q3: An agent workflow needs to loop through think→act→observe cycles until it has an answer, with a maximum of 10 iterations. What orchestration pattern is this?**

- A) Sequential chain
- B) DAG workflow
- C) Cycle with exit condition
- D) Parallel fan-out

**Answer: C** -This is a cyclic workflow (loop) with an exit condition (has answer or max iterations reached).

**Q4: Why is checkpointing critical for stateful orchestration?**

- A) It makes the agent faster
- B) It reduces token usage
- C) It enables fault tolerance, human-in-the-loop, and audit trails
- D) It improves the LLM's reasoning ability

**Answer: C** -Checkpointing saves state at each step, enabling resume after crashes, human review of decisions, and complete audit trails.

**Q5: A logic tree for a loan approval agent has three main branches: eligibility, financial risk, and policy compliance. If eligibility fails, the agent should immediately deny without checking the other branches. Which pattern does this represent?**

- A) Parallel execution with merge
- B) Sequential with early exit (short-circuit)
- C) Peer-to-peer collaboration
- D) Pub/sub event handling

**Answer: B** -This is a sequential evaluation with early exit (short-circuit). No need to evaluate further branches if a prerequisite fails.

---

## Key Takeaways for NVIDIA Certification

1. Prompt chains break complex tasks into focused steps: sequential, conditional, parallel, or transform
2. LangGraph is the primary framework for stateful orchestration using state machines
3. State is a typed dictionary shared across all nodes, persisted via checkpointing
4. Conditional edges enable routing, branching, and loops
5. Logic trees structure complex decisions into hierarchical sub-decisions
6. Cycles (loops) are essential for ReAct and iterative refinement patterns
7. Checkpointing enables fault tolerance, human-in-the-loop, and audit trails
8. DAG workflows handle complex dependencies with parallel and sequential steps

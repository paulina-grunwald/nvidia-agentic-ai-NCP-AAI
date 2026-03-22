# Planning Strategies and Adaptive Reasoning (Domain 5.3, 5.5)

## Overview

While the reasoning patterns (ReAct, CoT, ToT) in Domain 01 show *how* agents think, this section covers the *strategic layer* above that: how agents plan complex tasks, adapt when things fail, and improve their strategies through experience. This is where cognition becomes truly agentic.

Think of it this way:
- **Reasoning patterns** (01) = the algorithm for one step
- **Planning strategies** (05) = how to coordinate multiple steps over time
- **Adaptive reasoning** (05) = how to learn which strategy works best for which situation

---

## Part 1: Planning Strategies

### 1.1 Hierarchical Task Planning

The simplest planning approach breaks a goal into a tree of subgoals and actions. Each level down adds specificity.

#### Structure

```
Goal Level 0: "Deploy ML model to production"
├─ Subgoal 1: Prepare infrastructure
│  ├─ Action 1a: Provision GPU cluster
│  ├─ Action 1b: Set up monitoring
│  └─ Action 1c: Configure networking
├─ Subgoal 2: Validate model
│  ├─ Action 2a: Run test suite
│  ├─ Action 2b: Check accuracy metrics
│  └─ Action 2c: Verify performance SLAs
└─ Subgoal 3: Execute deployment
   ├─ Action 3a: Push container image
   ├─ Action 3b: Roll out gradual (10% → 50% → 100%)
   └─ Action 3c: Monitor for errors
```

#### When to Use

- Goals with clear dependencies (some steps must complete before others)
- Decomposable problems (each subgoal is independent once dependencies met)
- Tasks where you can pre-plan all steps

#### Limitations

- Rigid - if step 1a fails, the plan may become invalid
- Doesn't adapt to runtime discoveries
- Requires full upfront clarity on what needs to happen

---

### 1.2 Plan-and-Execute Pattern

Rather than fully planning upfront, plan-and-execute generates a plan, executes steps incrementally, re-evaluates after each step, and replans if needed.

#### Flow

```
Input: Goal ("Debug why inference latency increased")
  ↓
Plan Generation: "Step 1: Profile GPU. Step 2: Check memory. Step 3: Test batch sizes."
  ↓
Execute Step 1: [Call profiling tool] → Observation: "GPU utilization 95%"
  ↓
Evaluate: "Observation matches expectation?" → Yes, continue
  ↓
Execute Step 2: [Check memory] → Observation: "Memory bandwidth saturated"
  ↓
Replan?: "Based on Step 2, does original plan still work?" → Modified plan
  ↓
Execute Step 3 (modified): Test smaller batch sizes
  ↓
Final Answer: "Root cause: memory bandwidth. Solution: reduce batch size or use quantization."
```

#### Key Insight

The agent **generates a plan as a sequence of steps**, but continuously validates:
1. Does each step's observation match expectations?
2. Does the original plan still make sense given new information?
3. Should we replan?

#### Code Example (Conceptual)

```python
class PlanAndExecuteAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def run(self, goal: str):
        # Step 1: Generate plan
        plan = self.llm.generate_plan(goal)
        # Example: ["profile GPU", "check memory", "test batch sizes"]

        observations = []
        for step in plan:
            # Step 2: Execute step
            result = self.tools.execute(step)
            observations.append((step, result))

            # Step 3: Validate and potentially replan
            if self.should_replan(observations, plan):
                print(f"Replanning after step '{step}'...")
                plan = self.llm.generate_plan(goal, observations)

        # Step 4: Generate final answer
        answer = self.llm.synthesize_answer(goal, observations)
        return answer

    def should_replan(self, observations, original_plan):
        # Check if latest observation contradicts plan
        # or if we've discovered something that changes strategy
        return self.llm.evaluate_contradiction(observations, original_plan)
```

#### When to Use

- Exploratory problems with uncertain paths
- Goals where you discover new constraints as you go
- When tool feedback is likely to change your strategy
- RAG agents retrieving documents (plan might change based on first retrieval)

---

### 1.3 Reactive vs. Deliberative Planning

```
Deliberative (Plan-first):
Goal → [PLAN] → Execute Plan Step 1 → Step 2 → Step 3 → Answer

Reactive (Act-first):
Goal → [Reason] → [Act] → [Reason] → [Act] → Answer
```

| Aspect | Deliberative | Reactive |
|--------|-------------|----------|
| Planning overhead | High (cost upfront) | Low (inline with execution) |
| Adaptability | Lower (plan is fixed) | Higher (reacts to each observation) |
| Predictability | High (know path in advance) | Lower (discovery-based) |
| Best for | Structured, well-understood problems | Exploratory, uncertain domains |
| Example | "Deploy model: step 1→2→3" | "Debug issue: assess → try → reassess" |

**ReAct is inherently reactive** - it makes decisions as it learns. Deliberative planning generates a full plan upfront.

---

### 1.4 Adaptive Replanning

When execution reveals that your plan is no longer valid, you replan. This is the core mechanism for robust agents.

#### Example: Customer Support Agent

```
Original Plan:
1. Look up customer account
2. Check order history
3. Process refund
4. Send confirmation

Execution:
Step 1: Look up customer account → Found (account valid)
Step 2: Check order history → "Order is from 2019, outside 30-day window"
Step 3: BLOCKED - standard refund policy won't apply

Replan:
→ New Step 3a: Check customer loyalty (repeat customer?)
→ New Step 3b: If loyal, escalate for manager approval
→ New Step 3c: If approved, process refund with exception flag

Execute new plan...
```

#### Replan Triggers

1. **Constraint violation**: A precondition for the next step failed
2. **Goal conflict**: Two parts of the plan now conflict
3. **New option discovered**: A tool result showed a better path
4. **Resource limit**: Running low on tokens, time, or budget

---

## Part 2: Adaptive Reasoning

### 2.1 Memory-Augmented Reasoning

Adaptive agents don't just reason in the moment - they learn from past experiences and apply those lessons.

#### The Loop

```
Task N: Given problem, retrieve similar past tasks → Select proven strategy → Execute
                                                           ↓
                                                    Execute with that strategy
                                                           ↓
                                                    Success? Save experience
                                                           ↓
Task N+1: Can now retrieve Task N as precedent
```

#### Practical Example

An agent answering customer questions can build a memory of:

```json
{
  "past_experiences": [
    {
      "query": "How do I reset my password?",
      "strategy_used": "direct_answer (CoT)",
      "success": true,
      "tokens_used": 150
    },
    {
      "query": "I can't login and my password reset didn't work",
      "strategy_used": "direct_answer (CoT)",
      "success": false,
      "reason": "too_complex_for_reasoning"
    },
    {
      "query": "I can't login and my password reset didn't work",
      "strategy_used": "react_with_tools",
      "success": true,
      "tokens_used": 400
    }
  ]
}
```

When facing a new complex login issue, the agent can:
1. Search past_experiences for similar queries
2. See that CoT failed but ReAct succeeded for similar issues
3. Choose ReAct as the strategy

#### Implementation Sketch

```python
class MemoryAugmentedAgent:
    def __init__(self, llm, tools, memory):
        self.llm = llm
        self.tools = tools
        self.memory = memory  # Vector DB of past experiences

    def reason_about_task(self, task):
        # Step 1: Retrieve similar past tasks
        similar_tasks = self.memory.search(task, k=3)

        # Step 2: Extract what strategies worked
        successful_strategies = [
            t["strategy"] for t in similar_tasks
            if t.get("success")
        ]

        # Step 3: Use most common successful strategy
        strategy = self._vote_strategies(successful_strategies)

        # Step 4: Execute with that strategy
        result = self._execute_with_strategy(task, strategy)

        # Step 5: Save this experience for future
        self.memory.add({
            "task": task,
            "strategy": strategy,
            "success": result["success"],
            "tokens": result["tokens"],
            "duration": result["duration"]
        })

        return result
```

---

### 2.2 Feedback-Driven Strategy Adaptation

Agents should adapt their reasoning strategy based on real-time feedback.

#### Decision Tree for Strategy Selection

```
Task arrives → Estimate complexity using heuristics

Complexity = LOW (e.g., "What time is it?")
└─ Strategy: Direct answer or simple CoT
   └─ If fails: escalate to ReAct

Complexity = MEDIUM (e.g., "What's in this document?")
└─ Strategy: CoT or ReAct with tools
   └─ If CoT fails: switch to ReAct
   └─ If ReAct fails: Self-Reflection + retry

Complexity = HIGH (e.g., "Design a system for X with constraints Y and Z")
└─ Strategy: Start with ReAct
   └─ If stuck: escalate to ToT (explore alternatives)
   └─ If still stuck: Self-Reflection (critique own reasoning)
```

#### Concrete Adaptation Rules

| Situation | Original Strategy | Adapted Strategy | Why |
|-----------|-------------------|-----------------|-----|
| CoT trying to answer "What's my account balance?" | Reason it out | Switch to ReAct + tool | Can't reason without data |
| ReAct tool call fails 3 times | Keep retrying same tool | Try different tool | Tool not available/wrong |
| Complex creative task, CoT generated mediocre answer | Accept it | Switch to ToT | Need to explore alternatives |
| Hallucination detected (fact contradicts retrieval) | Accept answer | Self-Reflection + revise | Quality control |
| Approaching token limit, task not done | ReAct | Fallback to CoT summary | Cost control |

#### Code Example: Adaptive Strategy Switcher

```python
class AdaptiveReasoningAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.attempt_count = {}

    def solve(self, task):
        strategy = self._select_initial_strategy(task)
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                result = self._execute(task, strategy)
                if self._validate_result(result, task):
                    return result
            except Exception as e:
                # Execution failed or result invalid
                strategy = self._adapt_strategy(
                    original=strategy,
                    error=e,
                    attempt=attempt
                )
                print(f"Switched strategy to: {strategy}")

        raise Exception("All strategies exhausted")

    def _adapt_strategy(self, original, error, attempt):
        """Pick a new strategy based on failure."""
        if original == "direct":
            return "cot"  # Try reasoning
        elif original == "cot":
            return "react"  # Try with tools
        elif original == "react":
            return "tot"  # Try exploring multiple paths
        else:
            return "self_reflection"  # Last resort: critique
```

---

### 2.3 Meta-Reasoning: Reasoning About Reasoning

Meta-reasoning is when the agent decides which reasoning strategy to use. This is different from just using a strategy - the agent actively chooses and justifies the choice.

#### Example

```
Task: "Help me organize my research notes"

Agent reasoning about the task:
- This is a creative task (organizing without clear rules)
- There are multiple valid approaches (by date, by topic, by importance)
- The user might refine what they want (discovering preferences as we go)
- Therefore: Use ToT to explore organizational schemes, then Self-Reflection
  to refine based on user feedback

Agent to user: "I see this is a creative organization task. Let me explore
multiple approaches (by date, topic, and importance) and then we can refine
based on what feels right."
```

#### Implementation

```python
class MetaReasoningAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def solve(self, task, user_guidance=""):
        # Step 1: Analyze the task characteristics
        characteristics = self.llm.analyze_task(task)
        # Example: {
        #   "is_exploratory": True,
        #   "has_multiple_solutions": True,
        #   "requires_tools": False,
        #   "quality_critical": False,
        #   "time_sensitive": False
        # }

        # Step 2: Use characteristics to choose strategy
        strategy = self._select_strategy(characteristics)
        # Example: "tot" if exploratory, "self_reflection" if quality_critical

        # Step 3: Explain choice to user (transparency)
        explanation = self.llm.explain_strategy_choice(
            strategy,
            characteristics,
            task
        )
        print(f"Using {strategy}: {explanation}")

        # Step 4: Execute with chosen strategy
        result = self._execute(task, strategy)
        return result

    def _select_strategy(self, characteristics):
        """Map task characteristics to optimal strategy."""
        if characteristics["is_exploratory"] and characteristics["has_multiple_solutions"]:
            return "tot"
        elif characteristics["quality_critical"]:
            return "self_reflection"
        elif characteristics["requires_tools"]:
            return "react"
        else:
            return "cot"
```

---

### 2.4 Experience Replay

An agent can learn by "replaying" successful past experiences to guide new tasks.

#### Concept

```
When Agent solves Task A successfully:
→ Save: {goal, steps_taken, tools_used, reasoning, outcome}

When Agent later faces similar Task B:
→ Retrieve Task A's successful trajectory
→ Use it as an in-context example or constraint
→ "Here's how a similar task was solved... apply similar reasoning"
```

#### Example: Document Processing Agent

```
Past successful experience (saved):
- Goal: Extract entities from a messy email
- Steps: [Parse email → Clean text → Split by sections → Extract names/dates]
- Tools: text_parser, entity_extractor
- Reasoning: "Email format is common, use section splitting first"

New task: Extract entities from another messy email

Approach 1 (without replay):
- Agent tries random approach, fails, retries, eventually succeeds
- Took 5 attempts, 500 tokens

Approach 2 (with experience replay):
- Agent retrieves past successful trajectory
- Follows similar steps: [Parse → Clean → Split → Extract]
- Succeeds on first attempt, 100 tokens
```

#### LangGraph Implementation Pattern

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import PostgresSaver

# Store successful trajectories as checkpoints
def experience_replay(state, saved_experience):
    """
    Reuse successful steps from past experience.
    """
    if state.get("task_type") == saved_experience["task_type"]:
        # Apply same tool sequence
        return {
            "next_tool_to_use": saved_experience["tools"][0],
            "reasoning_hint": saved_experience["reasoning"],
            "is_replaying": True
        }

# In actual execution:
if agent_state.get("is_replaying"):
    # Follow the replay path, but allow deviations if needed
    result = execute_tool_from_experience()
else:
    # Discover new approach
    result = execute_dynamic()
```

---

### 2.5 Confidence-Based Escalation

As an agent executes, it should track confidence in its decisions. Low-confidence situations should escalate to humans or more expensive reasoning.

#### Confidence Levels

```
HIGH (>85%): "I'm very sure of this answer"
├─ Action: Return answer directly
├─ Cost: Low (1 LLM call)
└─ Example: "What is 2+2?" → 4

MEDIUM (50-85%): "I'm somewhat confident"
├─ Action: Verify with additional reasoning or tool
├─ Cost: Medium (2-3 LLM calls or tool)
└─ Example: "Recommend a product for someone" → verify with RAG

LOW (<50%): "I'm not sure"
├─ Action: Escalate to more expensive strategy or human
├─ Cost: High (ToT, Self-Reflection, or human review)
└─ Example: "Complex negotiation scenario" → escalate to human
```

#### Implementation

```python
class ConfidenceBasedAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def solve_with_escalation(self, task):
        # Generate initial answer
        answer, initial_confidence = self._generate_answer(task)

        if initial_confidence >= 0.85:
            # HIGH confidence: return directly
            return {
                "answer": answer,
                "confidence": initial_confidence,
                "escalated": False
            }
        elif initial_confidence >= 0.50:
            # MEDIUM confidence: verify
            verified_answer = self._verify_with_tools(answer, task)
            return {
                "answer": verified_answer,
                "confidence": 0.9,  # Verification increases confidence
                "escalated": False
            }
        else:
            # LOW confidence: escalate
            escalated_answer = self._escalate_to_expert(task)
            return {
                "answer": escalated_answer,
                "confidence": 0.95,
                "escalated": True,
                "reason": "low_initial_confidence"
            }

    def _generate_answer(self, task):
        """Generate answer and estimate confidence."""
        response = self.llm.generate(task)
        # Use LLM's own assessment or semantic similarity metrics
        confidence = self.llm.estimate_confidence(task, response)
        return response, confidence
```

---

## Part 3: LangGraph Plan-and-Execute Implementation

Here's a practical production example using LangGraph:

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import PostgresSaver
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

class PlanExecuteState(TypedDict):
    goal: str
    plan: list[str]  # List of planned steps
    observations: list[tuple[str, str]]  # (step, observation) pairs
    current_step_index: int
    reasoning_history: Annotated[list, operator.add]
    should_replan: bool
    final_answer: str

class PlanAndExecuteGraph:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def build_graph(self):
        graph = StateGraph(PlanExecuteState)

        # Node 1: Generate Plan
        def generate_plan(state):
            plan = self.llm.generate_plan(state["goal"])
            return {
                "plan": plan,
                "current_step_index": 0,
                "observations": []
            }

        # Node 2: Execute Current Step
        def execute_step(state):
            if state["current_step_index"] >= len(state["plan"]):
                return {"should_replan": False}

            step = state["plan"][state["current_step_index"]]
            result = self.tools.execute(step)

            new_observations = state["observations"] + [(step, result)]

            return {
                "observations": new_observations,
                "current_step_index": state["current_step_index"] + 1,
                "reasoning_history": [f"Executed: {step} → {result}"]
            }

        # Node 3: Check if replanning needed
        def check_replan(state):
            needs_replan = self.llm.should_replan(
                state["goal"],
                state["plan"],
                state["observations"]
            )
            return {"should_replan": needs_replan}

        # Node 4: Replan if needed
        def replan(state):
            new_plan = self.llm.generate_plan(
                state["goal"],
                state["observations"]
            )
            return {
                "plan": new_plan,
                "current_step_index": 0,
                "reasoning_history": ["Replanned based on observations"]
            }

        # Node 5: Generate final answer
        def final_answer(state):
            answer = self.llm.synthesize_answer(
                state["goal"],
                state["observations"]
            )
            return {"final_answer": answer}

        # Add nodes
        graph.add_node("generate_plan", generate_plan)
        graph.add_node("execute_step", execute_step)
        graph.add_node("check_replan", check_replan)
        graph.add_node("replan", replan)
        graph.add_node("final_answer", final_answer)

        # Add edges
        graph.set_entry_point("generate_plan")
        graph.add_edge("generate_plan", "execute_step")
        graph.add_edge("execute_step", "check_replan")

        # Conditional edge: replan or continue
        def should_replan_conditional(state):
            return "replan" if state.get("should_replan") else "execute_step"

        graph.add_conditional_edges(
            "check_replan",
            should_replan_conditional,
            {"replan": "replan", "execute_step": "execute_step"}
        )

        # Loop until all steps done
        def all_steps_done(state):
            return state["current_step_index"] >= len(state["plan"])

        graph.add_conditional_edges(
            "execute_step",
            lambda s: "final_answer" if all_steps_done(s) else "check_replan"
        )

        graph.add_edge("replan", "execute_step")
        graph.add_edge("final_answer", END)

        # Compile with checkpointing
        checkpointer = PostgresSaver("postgresql://...")
        return graph.compile(checkpointer=checkpointer)

# Usage
agent = PlanAndExecuteGraph(llm, tools)
compiled_graph = agent.build_graph()

result = compiled_graph.invoke(
    {
        "goal": "Find the top 3 productivity tools for remote teams",
        "plan": [],
        "observations": [],
        "current_step_index": 0,
        "reasoning_history": [],
        "should_replan": False,
        "final_answer": ""
    },
    config={"configurable": {"thread_id": "task_123"}}
)

print(result["final_answer"])
```

---

## Part 4: Integrated Example - Customer Support Agent

Let's tie everything together with a realistic example:

```python
class AdaptiveCustomerSupportAgent:
    """
    Demonstrates all adaptive reasoning techniques:
    - Hierarchical planning
    - Plan-and-execute with replanning
    - Memory-augmented reasoning
    - Feedback-driven adaptation
    - Confidence-based escalation
    """

    def __init__(self, llm, tools, memory):
        self.llm = llm
        self.tools = tools
        self.memory = memory

    def handle_inquiry(self, customer_query):
        """Main entry point for customer inquiry."""

        # Step 1: Retrieve similar past cases
        similar_cases = self.memory.search(customer_query, k=3)
        successful_approach = self._extract_successful_approach(similar_cases)

        # Step 2: Generate initial plan
        plan = self.llm.generate_plan(customer_query)

        # Step 3: Execute plan with adaptive replanning
        for step_num, step in enumerate(plan):
            result = self.tools.execute(step)

            # Confidence check
            confidence = self.llm.estimate_confidence(step, result)

            if confidence < 0.5:
                # Low confidence: escalate this step
                print(f"Low confidence on step {step_num}. Escalating to specialist.")
                result = self.tools.escalate_to_specialist(step, result)

            # Check if we need to replan
            if self.llm.should_replan([result]):
                print("Replanning based on new information...")
                plan = self.llm.generate_plan(
                    customer_query,
                    observations=[(step, result)]
                )

        # Step 4: Generate response
        response = self.llm.generate_response(customer_query, plan)

        # Step 5: Save experience for future
        self.memory.add({
            "query": customer_query,
            "plan_used": plan,
            "approach": successful_approach,
            "success": True,
            "response": response
        })

        return response
```

---

## Exam-Style Questions

**Q1: Plan-and-Execute vs. Hierarchical Planning**

An agent needs to debug a production ML inference service. Which strategy is best and why?

A) Hierarchical Planning - create a complete debug checklist upfront
B) Plan-and-Execute - generate a plan, execute steps, replan based on findings
C) Pure ReAct - just reason and act without planning
D) Self-Reflection - generate one answer then critique it

**Answer: B**. Debugging is inherently exploratory. You discover problems as you investigate. Plan-and-Execute generates an initial plan but adapts when findings change strategy.

---

**Q2: Confidence-Based Escalation**

A customer support agent is answering "What's my account balance?" The initial response has 30% confidence. What should the agent do?

A) Return the answer directly (low cost)
B) Use Self-Reflection to critique the answer
C) Escalate to a human specialist
D) Verify with a tool query to the database

**Answer: D**. This is a factual query that can be verified via tool. A database lookup gives 95%+ confidence. Only escalate to human if tools are unavailable.

---

**Q3: Memory-Augmented Reasoning**

An agent successfully solved a similar task yesterday using ReAct + tool A. Today it faces an analogous task but tries CoT first, which fails. What should happen?

A) Try CoT again (maybe it will work this time)
B) Suggest the human manually try tool A
C) Retrieve yesterday's experience and try ReAct + tool A
D) Switch to a completely different strategy

**Answer: C**. This is the definition of memory-augmented reasoning. Past successful experiences should inform strategy selection. The agent should recognize the similarity and reuse the proven approach.

---

**Q4: Adaptive Replanning Trigger**

During plan execution, the agent completes steps 1-3 of a 5-step plan. Step 4 requires customer approval, but feedback indicates the customer changed their requirements. When should replanning occur?

A) After step 3, before attempting step 4
B) After step 4 fails, then replan
C) Replan all 5 steps from the beginning
D) Continue with original plan; replanning isn't needed

**Answer: A**. Replanning should be triggered proactively when you discover constraints have changed. Waiting until step 4 fails is wasteful. This is the core of adaptive planning.

---

## Key Takeaways

1. **Hierarchical Planning** works for well-understood sequential tasks but requires replanning for exploratory work.

2. **Plan-and-Execute** is the practical production pattern: generate a plan, execute incrementally, replan when evidence contradicts your assumptions.

3. **Adaptive Reasoning** means selecting strategies based on task characteristics (exploratory vs. factual, time-sensitive vs. quality-critical, etc.).

4. **Memory-augmented reasoning** lets agents learn from past experiences and reuse successful approaches, dramatically reducing cost and improving speed.

5. **Confidence tracking** enables smart escalation: low-confidence decisions get more reasoning or human review, while high-confidence ones return directly.

6. **Meta-reasoning** (reasoning about which reasoning to use) is how agents become truly intelligent - they don't just follow pre-set strategies, they actively choose based on the problem at hand.

7. **Experience replay** and feedback loops create continuous improvement: every task is logged, successful patterns are identified and reused.

---

## Related Concepts

- **Reasoning Patterns (Domain 01)**: CoT, ReAct, ToT, Self-Reflection are the building blocks
- **State Management (Domain 01)**: Checkpointing enables fault tolerance during planning
- **Stateful Orchestration (Domain 01)**: Coordinates multiple agents' plans
- **Memory Management (Domain 01)**: Stores past experiences for retrieval
- **LangGraph Graphs (Domain 02)**: Implements planning workflows
- **Tool Integration (Domain 06)**: Tools provide the actions in executed plans

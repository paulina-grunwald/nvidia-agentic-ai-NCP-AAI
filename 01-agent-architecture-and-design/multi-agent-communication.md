# Multi-Agent Communication and Orchestration

## Overview

When multiple agents work together, they need ways to talk to each other and coordinate who does what. This file covers communication protocols (how agents exchange information) and orchestration patterns (how work gets divided and managed).

---

## Agent-to-Agent Communication Protocols

### 1. Message Passing (Direct)

The simplest model: one agent sends a message directly to another agent.

```
Agent A ---message---> Agent B
Agent B ---response---> Agent A
```

How it works:

- Each agent has an inbox/outbox
- Messages are structured (usually JSON with role, content, metadata)
- Synchronous (wait for reply) or asynchronous (fire and forget)

Pros: Simple, easy to debug, clear sender/receiver
Cons: Doesn't scale well when many agents need to communicate with each other (N×N connections)

Example in AutoGen:

```python
# Agent A sends to Agent B directly
agent_a.send(message="Please analyze this data", recipient=agent_b)
# Agent B processes and replies
```

### 2. Shared Memory / Blackboard System

All agents read from and write to a shared space. No direct agent-to-agent messages.

```
Agent A ──write──→ ┌──────────────┐ ←──read── Agent C
                   │  BLACKBOARD   │
Agent B ──write──→ │  (shared      │ ←──read── Agent D
                   │   state)      │
                   └──────────────┘
```

How it works:

- Central data store (database, in-memory dict, Redis, etc.)
- Agents check the blackboard for new information relevant to them
- Agents post their results back to the blackboard
- A controller may coordinate who reads/writes when

Pros: Decoupled agents (they don't need to know about each other), good for loosely coordinated tasks
Cons: Potential race conditions, harder to trace causality, needs concurrency management

This is how LangGraph manages state. The `StateGraph` acts as a shared blackboard that all nodes read from and write to:

```python
class SharedState(TypedDict):
    messages: list           # shared conversation
    research_results: str    # written by researcher agent
    analysis: str           # written by analyst agent
    final_report: str       # written by writer agent
```

### 3. Publish-Subscribe (Pub/Sub)

Agents subscribe to topics they care about. When an agent publishes to a topic, all subscribers get the message.

```
Agent A publishes to "data_ready" topic
     ↓
┌──────────────────────┐
│   Message Broker      │
│   topic: "data_ready" │
└──────┬───────┬───────┘
       ↓       ↓
   Agent B   Agent C
   (subscribed to "data_ready")
```

How it works:

- Agents subscribe to topics (e.g., "new_order", "analysis_complete", "error")
- When something happens, the responsible agent publishes an event
- All subscribers get notified and can act independently
- Message broker handles routing (Kafka, RabbitMQ, Redis Pub/Sub)

Pros: Very scalable, agents can be added/removed without changing others, great for event-driven systems
Cons: Harder to debug (who sent what?), eventual consistency, message ordering challenges

### 4. Request-Response

Structured like an API call. One agent makes a request, the other returns a response.

```
Agent A: request(task="summarize", data=document)
         ↓
Agent B: response(summary="The document discusses...")
```

Similar to how tools work in ReAct, but between agents instead of agent-to-tool. Often used when one agent has a specific capability (e.g., code execution, image analysis) that others need.

### 5. A2A Protocol (Agent-to-Agent)

Google's Agent2Agent (A2A) protocol is an emerging open standard for agent interoperability. It defines a common way for agents from different frameworks to discover each other and communicate.

Key concepts:

- **Agent Card**: A JSON document describing an agent's capabilities, endpoint, and supported tasks
- **Task lifecycle**: Agents exchange messages through a standardized task flow (submit, process, complete)
- **Streaming support**: Real-time updates via SSE
- **Push notifications**: Webhook-based updates for async tasks

Why it matters: As agent ecosystems grow, agents built with different frameworks (LangGraph, CrewAI, AutoGen) need to interoperate. A2A provides that standard.

### Comparison Table

| Protocol         | Best For                                | Scalability | Complexity | Framework Example      |
| ---------------- | --------------------------------------- | ----------- | ---------- | ---------------------- |
| Message Passing  | Small teams, clear hierarchies          | Low-Medium  | Low        | AutoGen conversations  |
| Shared Memory    | Loosely coupled tasks, state sharing    | Medium      | Medium     | LangGraph StateGraph   |
| Pub/Sub          | Event-driven, large systems             | High        | High       | Kafka + agent services |
| Request-Response | Specific capabilities, tool-like agents | Medium      | Low        | Function calling       |
| A2A Protocol     | Cross-framework interoperability        | High        | Medium     | Google A2A             |

---

## Multi-Agent Orchestration Patterns

Orchestration is about who decides what each agent does and when.

### 1. Sequential (Pipeline)

Agents execute one after another. Output of agent N becomes input of agent N+1.

```
User Query → Agent A (research) → Agent B (analyze) → Agent C (write) → Final Output
```

When to use: Tasks with clear stages where each step depends on the previous one.
Example: Data pipeline (extract → transform → summarize).

In CrewAI, this is the `sequential` process:

```python
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential
)
```

### 2. Parallel (Fan-Out / Fan-In)

Multiple agents work simultaneously on different parts of a task, then results are combined.

```
                 ┌─→ Agent A (market research)  ──┐
User Query ──→   ├─→ Agent B (competitor analysis)──├──→ Merge → Final Output
                 └─→ Agent C (financial data)    ──┘
```

When to use: Independent subtasks that don't depend on each other.
Example: Research report where different agents gather different types of data.

Benefits: Faster total execution time (parallelism).
Challenge: Merging results from different agents into a coherent output.

### 3. Hierarchical (Manager-Worker)

A supervisor/manager agent delegates tasks to specialized worker agents and synthesizes their outputs.

```
                    ┌─────────────┐
                    │  Supervisor  │
                    │  (decides    │
                    │   who works) │
                    └──┬───┬───┬──┘
                       ↓   ↓   ↓
                    Agent Agent Agent
                    (code)(data)(docs)
```

When to use: Complex tasks where a central coordinator needs to decide which agent handles what, and may need to re-route based on results.
Example: Customer support where a router agent decides if the query goes to billing, technical, or sales agent.

In LangGraph, this is built using conditional edges:

```python
def supervisor(state):
    # LLM decides which agent to call next
    decision = llm.invoke("Which agent should handle this?")
    return decision  # routes to the appropriate node

graph.add_conditional_edges("supervisor", supervisor, {
    "researcher": "research_node",
    "coder": "code_node",
    "writer": "write_node",
    "FINISH": END
})
```

In AutoGen, this uses the GroupChat with a manager:

```python
groupchat = GroupChat(
    agents=[coder, analyst, critic],
    messages=[],
    max_round=10
)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
```

### 4. Peer-to-Peer (Collaborative)

Agents communicate directly without a central coordinator. Each agent decides when to speak and who to address.

```
Agent A ←──→ Agent B
  ↕              ↕
Agent C ←──→ Agent D
```

When to use: Brainstorming, debate, review cycles where agents critique each other's work.
Example: Code review where a developer agent writes code, a tester agent finds bugs, and a reviewer agent suggests improvements, all interacting freely.

AutoGen's multi-agent conversations naturally support this pattern:

```python
# Agents take turns in a group chat, deciding who speaks next
user_proxy.initiate_chat(
    manager,
    message="Let's discuss the best approach to optimize our inference pipeline"
)
```

### 5. Dynamic Orchestration

The orchestration pattern itself changes based on the task. A meta-agent or router decides which pattern to use.

```
Simple query    → Single agent (no orchestration needed)
Research task   → Sequential pipeline
Complex report  → Hierarchical with parallel subtasks
Debate/review   → Peer-to-peer
```

This is the most flexible but also the most complex to implement.

### Pattern Comparison

| Pattern      | Latency          | Complexity  | Best For                        | Framework                               |
| ------------ | ---------------- | ----------- | ------------------------------- | --------------------------------------- |
| Sequential   | High (serial)    | Low         | Clear pipelines                 | CrewAI sequential                       |
| Parallel     | Low (concurrent) | Medium      | Independent subtasks            | LangGraph fan-out                       |
| Hierarchical | Medium           | Medium-High | Task routing, complex workflows | LangGraph supervisor, AutoGen GroupChat |
| Peer-to-Peer | Variable         | High        | Debate, review, creative tasks  | AutoGen conversations                   |
| Dynamic      | Variable         | Very High   | Adaptive systems                | Custom routing logic                    |

---

## Task Delegation and Conflict Resolution

### Task Delegation Strategies

When a supervisor assigns work to agents, it needs to decide:

1. **Capability-based**: Assign based on what each agent can do (code agent gets coding tasks, data agent gets data tasks)
2. **Load-based**: Assign to the agent with the lightest workload
3. **Priority-based**: Critical tasks go to the most reliable agent
4. **Round-robin**: Distribute evenly across agents

### Conflict Resolution

When agents disagree (e.g., two agents produce conflicting analyses):

| Strategy                | How It Works                                    |
| ----------------------- | ----------------------------------------------- |
| **Voting**              | Multiple agents vote, majority wins             |
| **Supervisor override** | Manager agent makes final decision              |
| **Confidence-based**    | Agent with highest confidence score wins        |
| **Human escalation**    | When agents can't agree, ask the human          |
| **Debate + judge**      | Agents present arguments, a judge agent decides |

Example of voting:

```python
# Three agents analyze the same data independently
results = [agent_a.analyze(data), agent_b.analyze(data), agent_c.analyze(data)]
# Majority vote
final_answer = most_common(results)
```

---

## Coordinator Patterns in Detail

### The Router Pattern

A lightweight agent that only decides where to send requests, doesn't do actual work:

```
User Query → Router Agent → {
    "billing question" → Billing Agent
    "technical issue"  → Tech Support Agent
    "sales inquiry"    → Sales Agent
    "unknown"          → General Agent
}
```

The router can be:

- **LLM-based**: Use an LLM to classify the query and route it
- **Rule-based**: Simple keyword matching or regex
- **Classifier-based**: A fine-tuned classification model

### The Orchestrator-Worker Pattern

More sophisticated than simple routing. The orchestrator:

1. Breaks the task into subtasks
2. Assigns subtasks to workers
3. Monitors progress
4. Handles failures and retries
5. Combines results

```python
# Pseudocode for orchestrator
class Orchestrator:
    def run(self, task):
        subtasks = self.decompose(task)          # Break down
        assignments = self.assign(subtasks)       # Delegate
        results = self.execute(assignments)       # Run workers
        if any_failed(results):
            results = self.retry_or_reassign()    # Handle failures
        return self.combine(results)              # Merge
```

---

## Framework Implementations

### CrewAI: Role-Based Teams

CrewAI organizes agents as a "crew" with defined roles:

```python
researcher = Agent(
    role="Senior Research Analyst",
    goal="Find comprehensive data on the topic",
    backstory="You are an expert researcher...",
    tools=[search_tool, web_scraper]
)

writer = Agent(
    role="Content Writer",
    goal="Create compelling content from research",
    backstory="You are a skilled writer...",
    tools=[writing_tool]
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential  # or Process.hierarchical
)
```

Communication: Agents pass outputs sequentially via task results, or in hierarchical mode, a manager agent coordinates.

### AutoGen: Conversational Multi-Agent

AutoGen uses natural conversation as the communication medium:

```python
assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding"})

# Two agents have a conversation to solve a problem
user_proxy.initiate_chat(assistant, message="Write a Python script to analyze sales data")
```

Communication: Message passing through structured conversations. Agents take turns. GroupChat enables multi-agent discussions.

### LangGraph: Stateful Graph Orchestration

LangGraph models multi-agent workflows as state machines:

```python
from langgraph.graph import StateGraph

graph = StateGraph(AgentState)
graph.add_node("researcher", research_agent)
graph.add_node("analyst", analysis_agent)
graph.add_node("supervisor", supervisor_agent)

# Supervisor decides routing
graph.add_conditional_edges("supervisor", route_decision)
graph.add_edge("researcher", "supervisor")
graph.add_edge("analyst", "supervisor")
```

Communication: Shared state (blackboard pattern). All nodes read from and write to the same state object.

---

## Exam-Style Questions

**Q1: A multi-agent system has a research agent, analysis agent, and writing agent. Each depends on the output of the previous. Which orchestration pattern is most appropriate?**

- A) Parallel
- B) Sequential
- C) Peer-to-peer
- D) Pub/Sub

**Answer: B** -Sequential (pipeline) because each agent's output feeds into the next.

**Q2: Three agents produce conflicting results when analyzing the same dataset. What is the most robust conflict resolution strategy?**

- A) Always use the first agent's result
- B) Voting across all three agents
- C) Randomly pick one
- D) Re-run all agents with higher temperature

**Answer: B** -Majority voting across independent agents is the most robust conflict resolution approach.

**Q3: A customer service system needs to route incoming queries to specialized agents (billing, tech support, sales). Which pattern fits best?**

- A) Peer-to-peer
- B) Sequential pipeline
- C) Hierarchical with a router/supervisor
- D) Parallel fan-out

**Answer: C** -A router/supervisor agent classifies the query and delegates to the appropriate specialized agent.

**Q4: Which communication protocol is best suited for a large-scale event-driven multi-agent system where new agents may be added frequently?**

- A) Direct message passing
- B) Shared memory blackboard
- C) Publish-subscribe
- D) Request-response

**Answer: C** -Pub/Sub scales well and allows agents to be added/removed without changing existing agents. They just subscribe to relevant topics.

**Q5: In LangGraph, how do agents share information?**

- A) Direct message passing between nodes
- B) Through a shared state object (blackboard pattern)
- C) Via external message queue
- D) Through file system

**Answer: B** -LangGraph uses a shared `StateGraph` where all nodes read from and write to the same typed state dictionary.

---

## Key Takeaways for NVIDIA Certification

1. Five main communication protocols: message passing, shared memory/blackboard, pub/sub, request-response, and A2A
2. Five orchestration patterns: sequential, parallel, hierarchical, peer-to-peer, dynamic
3. LangGraph = shared state (blackboard). AutoGen = message passing (conversations). CrewAI = role-based teams (sequential or hierarchical).
4. The hierarchical/supervisor pattern is the most common for production agent systems
5. Conflict resolution: voting, supervisor override, confidence-based, or human escalation
6. The router pattern is lightweight orchestration for classifying and delegating queries
7. A2A protocol is the emerging standard for cross-framework agent interoperability

# Langchain/Langraph
## Key Concept: Stateful Graphs

Define agent workflows as **directed graphs** with nodes and edges.

```mermaid
graph TD
    Start([Start]) --> Classify[Classify]
    Classify --> Route{Route Decision}
    Route -->|Technical| AgentA["Agent A<br/>(Tech Handler)"]
    Route -->|General| AgentB["Agent B<br/>(General Handler)"]
    Route -->|Specialized| AgentC["Agent C<br/>(Specialist)"]
    AgentA --> End([End])
    AgentB --> End
    AgentC --> End
```

## Key Components

| Component | Purpose |
| --- | --- |
| **StateGraph** | Define the workflow structure |
| **Nodes** | Actions/steps in the workflow |
| **Edges** | Transitions between nodes |
| **Conditional Edges** | Branching based on state |

## Key Features

| Feature | Description |
| --- | --- |
| **State persistence** | Save/resume workflow state |
| **Conditional routing** | Branch based on conditions |
| **Cycles** | Support loops (retries, ReAct) |
| **Checkpoints** | Resume from saved state |

## Memory & State Management

LangGraph provides built-in support for maintaining conversation and execution history:

### Memory Types

| Type | Use Case |
| --- | --- |
| **Conversation Memory** | Track user-agent dialogue history |
| **Execution Memory** | Store intermediate results from nodes |
| **Custom Memory** | Application-specific state storage |

### Checkpoints

Checkpoints are **snapshots of graph state** at specific execution points:

- **Automatic Checkpointing**: Save state after each node execution
- **Resume Capability**: Restart workflows from any checkpoint without reprocessing
- **Persistence**: Backed by databases (SQLite, PostgreSQL) or custom implementations
- **Use Cases**: Long-running tasks, error recovery, debugging, human-in-the-loop workflows

```python
# Enable checkpointing with database persistence
from langgraph.checkpoint import MemorySaver
from langgraph.graph import StateGraph

memory = MemorySaver()
graph = StateGraph()
# ... add nodes and edges ...

# Compile with checkpoint support
app = graph.compile(checkpointer=memory)

# Execute with thread_id to enable checkpoint tracking
result = app.invoke(
    {"input": "user query"},
    config={"configurable": {"thread_id": "conversation_123"}}
)

# Resume from checkpoint
state_snapshot = app.get_state({"configurable": {"thread_id": "conversation_123"}})
```

## Best For

- Complex conditional workflows
- Workflows with branching logic
- Long-running processes
- Workflows that need checkpointing

## NVIDIA Integration

LangGraph integrates seamlessly with NVIDIA's inference and deployment stack:

| NVIDIA Component | LangGraph Integration |
| --- | --- |
| **NIMs** | Wrap NIM endpoints as tool nodes for LLM inference |
| **Triton Inference Server** | Call Triton models as nodes for embeddings/specialized inference |
| **NeMo** | Use NeMo models as agents or tools within graph workflows |
| **Evaluation & Tuning** | Use checkpoints to save evaluation states and reduce redundant expensive calls |

## Advanced Patterns

### State Schema

Define strongly-typed state for multi-agent workflows:

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    classification: str  # Agent type from classifier
    tool_calls: list  # Tracked tool invocations
    error_count: int  # Retry tracking
```

### Error Handling with Checkpoints

Use checkpoints for robust error recovery:

```python
def agent_node_with_retry(state):
    try:
        result = call_nim_endpoint(state["input"])
        return {"output": result, "error_count": 0}
    except Exception as e:
        error_count = state.get("error_count", 0) + 1
        if error_count >= 3:
            return {"error": str(e), "error_count": error_count}
        # Checkpoint saves state before retry
        return {"error": str(e), "error_count": error_count}
```

### Parallel Execution

Execute multiple agents concurrently:

```python
async def parallel_agents(state):
    import asyncio

    results = await asyncio.gather(
        agent_a_async(state),
        agent_b_async(state),
        agent_c_async(state)
    )
    return {"parallel_results": results}

# Use with async graph compilation
app = graph.compile(checkpointer=memory)
result = await app.ainvoke(state, config=config)
```

## Complete Example with Checkpoints & NVIDIA Integration

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import PostgresSaver
from typing import TypedDict, Annotated
import httpx

# Define typed state
class WorkflowState(TypedDict):
    input: str
    classification: str
    result: str
    error: str = ""

# Node: Classify input using NIM
def classify_with_nim(state: WorkflowState):
    response = httpx.post(
        "http://nim-endpoint:8000/v1/classify",
        json={"text": state["input"]}
    )
    classification = response.json()["type"]
    return {"classification": classification}

# Conditional routing based on classification
def route_to_agent(state: WorkflowState):
    if state["classification"] == "technical":
        return "tech_agent"
    return "general_agent"

# Agent nodes
def tech_agent(state: WorkflowState):
    # Process technical queries with Triton model
    result = call_triton_model(state["input"], model="tech-solver")
    return {"result": result}

def general_agent(state: WorkflowState):
    # Process general queries with NIM LLM
    result = call_nim_llm(state["input"])
    return {"result": result}

# Build graph with checkpointing
graph = StateGraph(WorkflowState)
graph.add_node("classify", classify_with_nim)
graph.add_node("tech_agent", tech_agent)
graph.add_node("general_agent", general_agent)

graph.add_conditional_edges("classify", route_to_agent)
graph.set_entry_point("classify")
graph.add_edge("tech_agent", END)
graph.add_edge("general_agent", END)

# Compile with PostgreSQL persistence
checkpointer = PostgresSaver("postgresql://user:pass@localhost/langgraph")
app = graph.compile(checkpointer=checkpointer)

# Execute with checkpoint tracking (enables resume on failure)
config = {"configurable": {"thread_id": "workflow_123"}}
result = app.invoke(
    {"input": "How do I optimize CUDA kernels?"},
    config=config
)
```

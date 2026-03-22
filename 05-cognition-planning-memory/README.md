# Domain 5: Cognition, Planning, and Memory (10%)

## Overview

This domain covers the core cognitive processes that enable agents to reason, plan, and maintain state across executions.

## Topics to Cover

- **Memory Types**
  - Conversation memory and dialogue history
  - Execution memory and intermediate results
  - Long-term memory and knowledge storage
  - Memory architectures and retrieval patterns

- **Planning Mechanisms**
  - Goal decomposition and task planning
  - Tree-of-thought planning
  - Hierarchical planning approaches
  - Plan verification and refinement

- **State Management**
  - State persistence and checkpointing
  - State transitions and workflows
  - Context management
  - Session handling

- **Cognition Patterns**
  - Reflection and self-evaluation
  - Learning from interactions
  - Error recovery and adaptation

## Materials in This Domain

- **[memory-and-planning.md](./memory-and-planning.md)** - Memory types, conversation patterns, checkpointing, reasoning pattern selection
- **[planning-and-adaptive-reasoning.md](./planning-and-adaptive-reasoning.md)** - Hierarchical planning, plan-and-execute pattern, adaptive reasoning, memory-augmented strategies, confidence-based escalation

## Related Materials

- [Reasoning Patterns](../01-agent-architecture-and-design/reasoning-patterns.md) - ReAct, Chain-of-Thought, Tree of Thoughts (building blocks)
- [Memory Management](../01-agent-architecture-and-design/memory-management.md) - Long-term memory, semantic storage
- [Stateful Orchestration](../01-agent-architecture-and-design/stateful-orchestration.md) - Multi-agent coordination
- [LangGraph Memory & State Management](../02-agent-development/langchain.md#memory--state-management) - Implementation details
- [4-Layer Agent Architecture](../01-agent-architecture-and-design/4-layer-agent-architecture.md) - Planning as Layer 2

## Study Plan

1. Start with memory-and-planning.md (memory types, basic reasoning patterns)
2. Study planning-and-adaptive-reasoning.md (strategic planning and learning)
3. Understand how plans integrate with state management (from Domain 01)
4. Review production examples for scaling cognition

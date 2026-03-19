# Overview

**AutoGen** is Microsoft's open-source framework for building **multi-agent conversational AI systems**. It enables multiple AI agents to collaborate, converse with each other, and work together to solve complex tasks.

**Official Documentation:** https://microsoft.github.io/autogen/

**GitHub:** https://github.com/microsoft/autogen

---

# Core Concepts

## What Makes AutoGen Different

AutoGen's key innovation is **multi-agent conversation** - instead of one LLM doing everything, you create specialized agents that talk to each other.

Think of it like a team meeting: you have a coder, a reviewer, a planner, and they collaborate through conversation to complete tasks.

## The Conversation-Centric Approach

In AutoGen, everything happens through **message passing** between agents. Agents:

- Send messages to each other
- Can be backed by LLMs, humans, or tools
- Have defined roles and system prompts
- Can execute code and use tools

---

# Agent Types (Critical for Exam)

## 1. ConversableAgent (Base Class)

The foundation class that all other agents inherit from. Provides core conversation capabilities.

## 2. AssistantAgent

An LLM-powered agent designed to assist with tasks. Typically plays the role of an AI assistant.

**Key characteristics:**

- Backed by an LLM (GPT-4, etc.)
- Can generate code suggestions
- Can reason about problems
- Does NOT execute code by default

## 3. UserProxyAgent

Represents human users or acts as a proxy that can execute code.

**Key characteristics:**

- Can execute code locally
- Can request human input (human-in-the-loop)
- Often paired with AssistantAgent
- Handles tool/function execution

## 4. GroupChatManager

Orchestrates conversations between multiple agents in a group chat setting.

**Key characteristics:**

- Manages turn-taking between agents
- Supports different speaker selection methods
- Enables complex multi-agent workflows

---


# Human-in-the-Loop Modes

The `human_input_mode` parameter controls when human intervention is requested:

| Mode | Behavior |
| --- | --- |
| `ALWAYS` | Ask human for input after every agent response |
| `TERMINATE` | Ask human only when termination condition is met |
| `NEVER` | Fully autonomous, no human input requested |

**Exam tip:** Know when to use each mode - `NEVER` for automation, `ALWAYS` for supervised tasks, `TERMINATE` for review before completion.


# Tool/Function Calling

Agents can use tools through function registration:

```python
from autogen import register_function

def get_weather(city: str) -> str:
    """Get current weather for a city"""
    return f"Weather in {city}: 72°F, Sunny"

# Register the function with agents
register_function(
    get_weather,
    caller=assistant,      # Agent that decides to call
    executor=user_proxy,   # Agent that executes
    description="Get the current weather for a city"
)
```

---


# Key Features Summary (Exam Focus)

| Feature | Description |
| --- | --- |
| Multi-Agent Conversations | Multiple agents collaborate through message passing |
| Flexible Agent Roles | AssistantAgent (LLM), UserProxyAgent (execution), Custom agents |
| Code Execution | Built-in ability to run generated code safely |
| Human-in-the-Loop | Configurable human intervention points |
| Tool Integration | Register custom functions as tools |
| Group Chat | Orchestrate complex multi-agent workflows |
| LLM Agnostic | Works with OpenAI, Azure, local models |

---

# AutoGen vs Other Frameworks

| Framework | Primary Focus |
| --- | --- |
| **AutoGen** | Multi-agent conversation and collaboration |
| **LangChain** | Chains, tools, and retrieval pipelines |
| **LlamaIndex** | Data indexing and RAG |
| **CrewAI** | Role-based agent teams (built on LangChain) |

<aside>
🔥

**Exam tip:** AutoGen's differentiator is its **conversation-first** approach to multi-agent systems.

</aside>

---

# Common Exam Patterns

**Q: When would you use AutoGen?**

A: When you need multiple specialized agents to collaborate on complex tasks through conversation (code review, research teams, planning + execution).

**Q: What's the difference between AssistantAgent and UserProxyAgent?**

A: AssistantAgent is LLM-powered for reasoning/generation. UserProxyAgent handles code execution and human interaction.

**Q: How do you ensure safe code execution in AutoGen?**

A: Use `use_docker=True` in `code_execution_config` for sandboxed execution.

---

# Key Terminology

- **Conversable Agent**: Any agent that can participate in conversations
- **Initiate Chat**: Method to start a conversation between agents
- **Termination Condition**: Conditions that end the agent conversation loop
- **Max Rounds**: Limit on conversation turns in group chat
- **LLM Config**: Configuration for the language model backing an agent

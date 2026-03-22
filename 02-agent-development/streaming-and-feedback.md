# Dynamic Conversation Flows, Streaming, and Feedback (Objective 2.5)

Developing dynamic conversation flows with real-time streaming and feedback mechanisms.

---

## Real-Time Streaming Patterns

### Why Streaming Matters

An LLM generating a 500-token response might take 3-5 seconds. Without streaming, the user sees nothing for 3-5 seconds, then the full response appears. With streaming, tokens appear as they're generated, starting within 200ms.

Streaming is essential for production agents because:
- Reduces perceived latency dramatically
- Lets users start reading immediately
- Enables early cancellation if the agent is going the wrong direction
- Required for voice agents (TTS needs text chunks to start speaking)

### Server-Sent Events (SSE)

SSE is the most common streaming protocol for LLM responses. It's a one-way channel from server to client over HTTP.

```
Client → HTTP POST /v1/chat/completions (stream=true)
Server → text/event-stream response:

data: {"choices": [{"delta": {"content": "The"}}]}
data: {"choices": [{"delta": {"content": " weather"}}]}
data: {"choices": [{"delta": {"content": " in"}}]}
data: {"choices": [{"delta": {"content": " Paris"}}]}
data: {"choices": [{"delta": {"content": " is"}}]}
data: {"choices": [{"delta": {"content": " sunny"}}]}
data: [DONE]
```

NVIDIA NIM uses SSE for streaming. When you set `stream=True`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://nim-endpoint:8000/v1", api_key="not-used")

stream = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[{"role": "user", "content": "Explain agentic AI"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Key details:
- Each chunk contains a small piece (often 1 token) of the response
- The `delta` field contains the incremental content (vs `message` for full responses)
- `[DONE]` signals the stream is complete
- SSE works over standard HTTP, no special infrastructure needed

### WebSockets

WebSockets provide full bidirectional communication. Both client and server can send messages at any time.

```
Client ←→ WebSocket Connection ←→ Server

Client sends: {"type": "message", "content": "What's the weather?"}
Server sends: {"type": "token", "content": "The"}
Server sends: {"type": "token", "content": " weather"}
Client sends: {"type": "cancel"}  ← user can interrupt!
Server sends: {"type": "cancelled"}
```

When to use WebSockets vs SSE:

| Feature | SSE | WebSocket |
|---|---|---|
| Direction | Server → Client only | Bidirectional |
| Protocol | HTTP | ws:// or wss:// |
| Reconnection | Built-in auto-reconnect | Manual reconnection |
| Complexity | Simple | More complex |
| Best for | Streaming LLM responses | Interactive agents (voice, real-time collaboration) |

For most LLM streaming, SSE is sufficient and simpler. Use WebSockets when you need the client to send messages during streaming (e.g., cancellation, voice barge-in).

### Streaming with Tool Calls

When an agent is using tools, streaming gets more complex. The agent might:
1. Stream reasoning text
2. Pause to make a tool call
3. Wait for tool result
4. Resume streaming the response

```
Stream: "Let me check the weather" [pause]
        → tool_call: get_weather("Paris")
        → tool_result: {temp: 22, condition: "sunny"}
Stream: "The weather in Paris is 22°C and sunny."
```

NVIDIA NIM and OpenAI handle this by streaming tool call chunks:

```python
# Tool calls come as streamed chunks too
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.tool_calls:
        # Accumulate tool call arguments
        tool_call_chunk = delta.tool_calls[0]
        # tool_call_chunk.function.arguments arrives in pieces
    elif delta.content:
        print(delta.content, end="")
```

---

## Streaming in Agent Frameworks

### LangGraph Streaming

LangGraph supports streaming at multiple levels:

```python
# Stream events from the graph
async for event in app.astream_events(
    {"messages": [("user", "Research NVIDIA stock")]},
    config=config,
    version="v2"
):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        # Token-by-token LLM output
        print(event["data"]["chunk"].content, end="")
    elif kind == "on_tool_start":
        print(f"\n[Calling tool: {event['name']}]")
    elif kind == "on_tool_end":
        print(f"\n[Tool result: {event['data'].content[:100]}]")
```

LangGraph's `astream_events` gives you visibility into every step: which node is executing, what the LLM is generating, when tools are called, what results come back.

### AutoGen Streaming

AutoGen supports streaming through its agent conversation:

```python
assistant = AssistantAgent(
    "assistant",
    llm_config={
        "config_list": [{"model": "gpt-4", "api_key": key}],
        "stream": True  # Enable streaming
    }
)
```

In AutoGen, streaming shows the conversation between agents in real-time. When agent A sends a message, you see it streaming as agent B generates its response.

---

## Dynamic Conversation Flow Management

### Conversation State Tracking

An agent needs to track where it is in a conversation and what information it has:

```python
class ConversationState:
    def __init__(self):
        self.current_intent = None     # What the user wants
        self.slots = {}                # Required info collected so far
        self.missing_slots = []        # Info still needed
        self.conversation_phase = "greeting"  # greeting, info_gathering, executing, confirming
```

### Slot Filling

For task-oriented agents (booking, ordering, support), the agent needs to collect specific pieces of information:

```
Agent task: Book a flight
Required slots: origin, destination, date, passengers

Turn 1: User: "I want to fly to Tokyo"
   → slots filled: {destination: "Tokyo"}
   → missing: [origin, date, passengers]

Turn 2: Agent: "Where will you be flying from?"
   User: "San Francisco"
   → slots filled: {destination: "Tokyo", origin: "San Francisco"}
   → missing: [date, passengers]

Turn 3: Agent: "When would you like to travel?"
   User: "Next Friday, just me"
   → slots filled: {destination: "Tokyo", origin: "SF", date: "March 28", passengers: 1}
   → missing: [] → proceed to booking
```

### Intent Switching

Users change their minds mid-conversation. The agent needs to handle this gracefully:

```
Turn 1: User: "Book me a flight to Tokyo"
   → intent: book_flight

Turn 2: Agent: "What dates?"
   User: "Actually, first tell me what the weather's like there"
   → intent changed: book_flight → get_weather
   → agent should: answer weather question, then return to booking
```

Implementation in LangGraph:
```python
def detect_intent_change(state):
    current_intent = classify_intent(state["messages"][-1])
    if current_intent != state["active_intent"]:
        # Stack the old intent so we can return to it
        return {
            "active_intent": current_intent,
            "intent_stack": state.get("intent_stack", []) + [state["active_intent"]]
        }
    return {}
```

### Conversation Recovery

When something goes wrong, the agent should recover gracefully:

```
Turn 1: User: "Transfer $500 to John"
Turn 2: Agent: "I'll transfer $500 to John's account ending in 4521. Confirm?"
Turn 3: User: "Wait, not John. I meant Jane."
   → Agent should: correct the recipient, re-confirm
Turn 4: Agent: "Got it, transferring $500 to Jane's account ending in 7832. Confirm?"
Turn 5: User: "Yes"
```

The agent needs to:
- Detect corrections ("not X, I meant Y")
- Update the relevant slot without losing other collected info
- Re-confirm before acting

---

## Feedback Mechanisms

### Explicit User Feedback

Structured feedback the user actively provides:

**Inline corrections**: User corrects the agent during conversation
```
Agent: "I found flights to Osaka for March 28."
User: "I said Tokyo, not Osaka."
Agent: "My apologies. Let me search for Tokyo instead."
```

**Ratings**: After the interaction
```
Agent: "Is there anything else I can help with?"
User: "No, that's all."
Agent: "How would you rate this interaction? [1-5 stars]"
```

**Thumbs up/down**: Per-response feedback
```
Agent: "Here are the top 3 results..." [👍] [👎]
```

### Implicit Feedback Signals

User behavior that indicates quality without explicit rating:

| Signal | Positive Indicator | Negative Indicator |
|---|---|---|
| Reformulation | N/A | User rephrases the same question |
| Follow-up | "Great, now also..." | "That's not what I asked" |
| Engagement | Continues conversation | Abandons conversation |
| Copy/use | User copies the response | User ignores the response |
| Regeneration | N/A | User clicks "try again" |
| Speed | Quick follow-up | Long pause then correction |

### Feedback Integration Pipeline

```
Conversation → Feedback (explicit + implicit)
                        ↓
               Logging / Analytics
                        ↓
              ┌─────────┴─────────┐
              ↓                   ↓
    Short-term: Adjust      Long-term: Fine-tune
    current session         evaluation datasets
    (prompt adaptation)     (model improvement)
```

Short-term adaptation: If a user corrects the agent, update the conversation context so the mistake doesn't repeat in this session.

Long-term improvement: Aggregate feedback across users to identify patterns (which tool calls fail? which questions get bad ratings?) and use for evaluation/fine-tuning.

### Feedback in NVIDIA Context

NVIDIA's approach to feedback in agent systems:
- **NeMo Guardrails** can log blocked/flagged interactions as negative feedback
- **Agent Intelligence Toolkit** supports evaluation pipelines that consume feedback data
- **NIM endpoints** can log inference requests and latency for performance feedback

---

## Async Patterns for Agents

### Why Async Matters

Agents make many external calls: LLM inference, tool calls, database queries. If these are synchronous (blocking), the agent is idle while waiting.

```python
# Synchronous - slow, blocks on each call
result_1 = tool_a(query)       # wait 2s
result_2 = tool_b(query)       # wait 2s
result_3 = llm_reason(result_1, result_2)  # wait 3s
# Total: 7s

# Async - parallel where possible
result_1, result_2 = await asyncio.gather(
    tool_a(query),             # 2s ─┐
    tool_b(query)              # 2s ─┤ parallel = 2s
)                              #     ┘
result_3 = await llm_reason(result_1, result_2)  # wait 3s
# Total: 5s
```

### Async Streaming with LangGraph

```python
async for chunk in app.astream(
    {"messages": [HumanMessage(content="Research this topic")]},
    config=config
):
    # Process each streamed chunk as it arrives
    for key, value in chunk.items():
        print(f"Node '{key}' produced output")
```

### Handling Long-Running Tools

Some tools take a long time (database migrations, file processing). Use async with status updates:

```python
async def long_running_analysis(data):
    yield {"status": "starting", "message": "Beginning analysis..."}

    # Phase 1
    cleaned = await clean_data(data)
    yield {"status": "progress", "message": "Data cleaned. Analyzing patterns..."}

    # Phase 2
    patterns = await find_patterns(cleaned)
    yield {"status": "progress", "message": f"Found {len(patterns)} patterns. Generating report..."}

    # Phase 3
    report = await generate_report(patterns)
    yield {"status": "complete", "result": report}
```

This gives the user progress updates instead of silence during long operations.

---

## Exam-Style Questions

**Q1: A user interacts with an agent and sees no response for 4 seconds, then the full answer appears. What technique would most improve the user experience?**
- A) Use a faster model
- B) Enable SSE streaming so tokens appear as they're generated
- C) Cache all responses
- D) Reduce the response length

**Answer: B** - SSE streaming shows tokens as they're generated, reducing perceived latency from 4 seconds to near-instant first token.

**Q2: An agent is collecting information for a flight booking. The user has provided destination and date but not origin. What conversation pattern is the agent using?**
- A) Intent classification
- B) Slot filling
- C) Prompt chaining
- D) Self-reflection

**Answer: B** - Slot filling is the pattern of collecting required pieces of information (slots) through conversation turns until all needed data is gathered.

**Q3: Which streaming protocol allows the client to send messages to the server during streaming (e.g., to cancel a response)?**
- A) SSE (Server-Sent Events)
- B) WebSocket
- C) HTTP polling
- D) gRPC unary

**Answer: B** - WebSockets are bidirectional, allowing the client to send cancel/interrupt signals while the server is still streaming. SSE is server-to-client only.

**Q4: A user says "Actually, I meant Jane not John" mid-conversation. What should the agent do?**
- A) Start the conversation over from scratch
- B) Update the relevant slot (recipient) and re-confirm without losing other collected information
- C) Ignore the correction
- D) Ask the user to start over

**Answer: B** - The agent should correct only the specific slot that changed while preserving all other collected information, then re-confirm.

**Q5: Which LangGraph method provides the most detailed streaming, including tool call events and per-node outputs?**
- A) `app.invoke()`
- B) `app.stream()`
- C) `app.astream_events()`
- D) `app.batch()`

**Answer: C** - `astream_events()` provides the most granular streaming, emitting events for model tokens, tool starts/ends, and node-level outputs.

---

## Key Takeaways for NVIDIA Certification

1. SSE is the standard streaming protocol for LLM responses (used by NVIDIA NIM)
2. WebSockets add bidirectional communication when the client needs to interrupt or send during streaming
3. Streaming with tool calls requires accumulating tool call chunks before executing
4. Slot filling is the key pattern for task-oriented conversation (booking, ordering)
5. Intent switching and conversation recovery handle real-world messy dialogue
6. Implicit feedback (reformulations, abandonment, regeneration) is as valuable as explicit ratings
7. Async patterns (asyncio.gather) parallelize independent tool calls for lower latency
8. LangGraph's `astream_events` gives fine-grained visibility into every step of agent execution

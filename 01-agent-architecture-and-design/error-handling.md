# Error Handling

## Why Error Handling Matters

Agents interact with external systems that can fail:

- API timeouts
- Rate limits
- Invalid responses
- Network errors
- Tool execution failures

### Error Handling Strategies
| Strategy | Description | When to Use |
| --- | --- | --- |
| **Retry with backoff** | Wait and retry | Transient failures |
| **Fallback tools** | Try alternative tool | Primary tool unavailable |
| **Graceful degradation** | Provide partial answer | Some data unavailable |
| **Human escalation** | Ask human for help | Critical failures |
| **Self-correction** | Agent fixes its own mistakes | Recoverable errors |


### Error Handling Best Practices

| Practice | Description |
| --- | --- |
| **Log all errors** | For debugging and monitoring |
| **Set timeouts** | Don't wait forever for tools |
| **Validate tool outputs** | Check responses make sense |
| **Limit retry attempts** | Avoid infinite loops |
| **Provide context in errors** | Help agent understand what went wrong |

---

## Circuit Breaker Pattern for Resilience

The **Circuit Breaker** pattern prevents cascading failures by monitoring error rates and temporarily stopping calls to a failing service when it exceeds a threshold. Critical for agentic AI systems that depend on external tools and APIs.

### How It Works

Circuit Breaker has three states:

| State | Behavior | Transition |
| --- | --- | --- |
| **CLOSED** | Normal operation - requests pass through, errors counted | Move to OPEN if error threshold exceeded |
| **OPEN** | Requests rejected/routed to fallback - no calls to failing service | After timeout, move to HALF-OPEN |
| **HALF-OPEN** | Limited test requests allowed to verify recovery | SUCCESS → CLOSED, FAILURE → OPEN |

### Why Resilience Matters for AI Agents

Agents operate in unpredictable environments where failures are inevitable:

- LLM API rate limits and timeouts
- Tool execution failures
- Network interruptions
- Malformed outputs (hallucinations, schema violations)
- Cascading failures in multi-agent systems
- Model drift over time

Production agents must be designed to **absorb failures without compromising task correctness or user trust**.

### Implementation Pattern

```javascript
// Pseudocode for circuit breaker
class CircuitBreaker:
  state = CLOSED
  failureCount = 0
  successCount = 0
  lastFailureTime = null
  threshold = 5        // failures before opening
  timeout = 60         // seconds before attempting half-open

  call(tool, input):
    if state == OPEN:
      if time.now() - lastFailureTime > timeout:
        state = HALF_OPEN
      else:
        throw CircuitBreakerOpenError

    try:
      result = tool(input)
      onSuccess()
      return result
    except Exception as e:
      onFailure()
      throw e

  onSuccess():
    failureCount = 0
    if state == HALF_OPEN:
      state = CLOSED

  onFailure():
    failureCount += 1
    lastFailureTime = time.now()
    if failureCount >= threshold:
      state = OPEN
```

### When to Apply Circuit Breaker

**Good candidates:**
- Unreliable external APIs (weather, search, databases)
- Rate-limited services
- Flaky network calls
- Multi-agent communication

**Not needed:**
- Local function calls
- Reliable internal services with SLAs > 99.95%
- Deterministic operations

### Combined with Other Strategies

- **Circuit Breaker + Retry**: Retry during CLOSED state, fail fast when OPEN
- **Circuit Breaker + Fallback**: When OPEN, route to alternative tool
- **Circuit Breaker + Adaptive Backoff**: Increase timeout as failures accumulate



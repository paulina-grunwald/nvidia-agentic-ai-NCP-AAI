# Observability and Tracing (8.1, 8.2)

Agent observability is about seeing what your agent is actually doing in production. Unlike traditional ML models that take input and produce output, agents make decisions, call tools, reason through problems, and sometimes fail in subtle ways. You need structured logging, distributed tracing, and real-time dashboards to understand agent behavior.

---

# What to Log in Agent Systems

The key insight: **log at decision boundaries**. Not every line of code, but every meaningful state transition.

## Structured Logging Strategy

| What to Log | Example | Why It Matters |
| --- | --- | --- |
| **Request entry** | `agent_start: query="what is the weather", user_id=123` | Trace a request through the entire system |
| **Tool calls** | `tool_call: name=web_search, args={query: "NVIDIA H100"}, timestamp=T1` | Understand agent decision-making |
| **Tool results** | `tool_result: name=web_search, output="H100 specs...", latency_ms=450` | See what information agent received |
| **Reasoning traces** | `reasoning: step=2, thought="I need more info", action="call search"` | Debug why agent made decisions |
| **Token usage** | `tokens_used: input=1200, output=450, total=1650, cost=$0.003` | Track costs per request |
| **Latency per step** | `step_latency: tool_call=450ms, inference=320ms, reasoning=50ms` | Find bottlenecks |
| **Errors and retries** | `error: type=tool_timeout, tool=api_call, retry_count=2, attempt=2of3` | Understand failure patterns |
| **Agent termination** | `agent_end: success=true, total_latency_ms=2300, tool_calls=3, final_answer=...` | Complete request picture |

## Logging Format: Structured JSON

Don't log plain strings. Use structured JSON so you can query and aggregate in observability systems.

```json
{
  "timestamp": "2026-03-22T14:32:15.123Z",
  "request_id": "req_abc123xyz789",
  "trace_id": "trace_xyz789abc123",
  "span_id": "span_001",
  "agent_name": "customer_support_agent",
  "level": "INFO",
  "event_type": "tool_call",
  "tool_name": "customer_database_lookup",
  "tool_args": {
    "customer_id": "cust_456"
  },
  "user_id": "user_123",
  "tokens_used": {
    "input": 1200,
    "output": 450
  },
  "model": "gpt-4-turbo",
  "model_version": "gpt-4-turbo-2025-02",
  "latency_ms": 450,
  "gpu_id": "gpu_2",
  "error": null
}
```

**Why structure matters:**
- Query: `event_type=tool_call AND tool_name=customer_database_lookup AND user_id=user_123`
- Aggregate: `avg(latency_ms) by tool_name` to find slow tools
- Alert: `if avg(error_rate) > 0.05 then page_oncall`

---

# Distributed Tracing with OpenTelemetry

A single user request might touch 5+ services:
1. User request → API gateway
2. Agent orchestrator
3. LLM inference (NIM)
4. Tool service 1 (web search)
5. Tool service 2 (database)
6. Response back to user

**Problem:** If the user says "it's slow", which service is the bottleneck?

**Solution:** Trace IDs that flow through every service.

## OpenTelemetry Setup for Agent Systems

```python
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger-collector",
    agent_port=6831,
)

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Auto-instrument HTTP libraries
RequestsInstrumentor().instrument()
HTTPXClientInstrumentor().instrument()

tracer = trace.get_tracer(__name__)
```

## Tracing an Agent Execution

```python
def run_agent(query: str, user_id: str):
    """Main agent loop - wrapped in a span."""

    with tracer.start_as_current_span("agent_execution") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("query", query)
        span.set_attribute("agent_name", "customer_support")

        # Step 1: LLM decision
        with tracer.start_as_current_span("llm_decision") as llm_span:
            llm_span.set_attribute("model", "gpt-4-turbo")
            llm_span.set_attribute("temperature", 0.7)

            decision = llm.generate(
                prompt=f"Query: {query}",
                model="gpt-4-turbo"
            )
            llm_span.set_attribute("decision", decision)

        # Step 2: Tool execution
        if decision.requires_tool_call:
            with tracer.start_as_current_span("tool_execution") as tool_span:
                tool_span.set_attribute("tool_name", decision.tool)
                tool_span.set_attribute("tool_args", decision.args)

                result = execute_tool(decision.tool, decision.args)

                tool_span.set_attribute("tool_result", result)
                tool_span.set_attribute("tool_latency_ms", result.latency_ms)

        # Step 3: Final response
        with tracer.start_as_current_span("final_response") as resp_span:
            resp_span.set_attribute("total_tool_calls", len(decision.history))
            resp_span.set_attribute("success", True)

            return final_answer

# Trace ID automatically propagates to logs
logger.info("Agent completed", extra={
    "trace_id": span.get_span_context().trace_id,
    "span_id": span.get_span_context().span_id
})
```

**Key benefit:** In Jaeger/Zipkin, you see a waterfall of all spans, instantly identifying where latency lives.

---

# Tracing Agent Decision Chains (ReAct)

ReAct agents think out loud: "Thought → Action → Observation → Thought → ...". You need to trace this chain.

## Capturing ReAct Traces

```python
from typing import List

class ReActTrace:
    """Capture the full decision chain."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.steps: List[dict] = []

    def add_thought(self, thought: str):
        """Agent's reasoning step."""
        self.steps.append({
            "step_number": len(self.steps) + 1,
            "type": "thought",
            "content": thought,
            "timestamp": datetime.now().isoformat()
        })
        logger.info("Agent thought", extra={
            "request_id": self.request_id,
            "thought": thought,
            "step": len(self.steps)
        })

    def add_action(self, tool_name: str, tool_args: dict):
        """Agent decided to call a tool."""
        self.steps.append({
            "step_number": len(self.steps) + 1,
            "type": "action",
            "tool": tool_name,
            "args": tool_args,
            "timestamp": datetime.now().isoformat()
        })
        logger.info("Agent action", extra={
            "request_id": self.request_id,
            "tool": tool_name,
            "args": tool_args
        })

    def add_observation(self, result: dict, latency_ms: int):
        """Result from the tool call."""
        self.steps.append({
            "step_number": len(self.steps) + 1,
            "type": "observation",
            "result_length": len(str(result)),
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat()
        })
        logger.info("Tool result received", extra={
            "request_id": self.request_id,
            "result_length": len(str(result)),
            "latency_ms": latency_ms
        })

    def to_dict(self) -> dict:
        """Export trace for storage/analysis."""
        return {
            "request_id": self.request_id,
            "total_steps": len(self.steps),
            "step_breakdown": {
                "thoughts": sum(1 for s in self.steps if s["type"] == "thought"),
                "actions": sum(1 for s in self.steps if s["type"] == "action"),
                "observations": sum(1 for s in self.steps if s["type"] == "observation")
            },
            "steps": self.steps
        }
```

## Example Trace in Observability Dashboard

```
Request: "What's NVIDIA's stock price and latest earnings?"

Step 1 [THOUGHT] 0.1s
  "I need to search for current stock price and earnings info"

Step 2 [ACTION] tool=web_search, args={query: "NVIDIA stock price"}

Step 3 [OBSERVATION] 0.45s
  Got 5 results about stock price ($120.50)

Step 4 [THOUGHT] 0.05s
  "Now I need earnings info"

Step 5 [ACTION] tool=web_search, args={query: "NVIDIA Q1 2026 earnings"}

Step 6 [OBSERVATION] 0.52s
  Got earnings data (revenue $65B)

Step 7 [THOUGHT] 0.08s
  "I have enough information to answer"

Step 8 [FINAL_ANSWER] 1.2s total latency
  "NVIDIA is trading at $120.50. Latest earnings Q1 2026: $65B revenue"
```

---

# Observability Stack Components

## Architecture Overview

```
Agent System
    ↓
Logging Layer (Structured JSON)
    ↙ ↓ ↘
Logs    Metrics    Traces
    ↓        ↓        ↓
ELK Stack  Prometheus  Jaeger/Zipkin
    ↓        ↓        ↓
Kibana   Grafana    Jaeger UI
         (central dashboard)
```

## Logs: ELK Stack (Elasticsearch, Logstash, Kibana)

**Why:**
- Store millions of structured log events
- Full-text search: `error:"timeout" AND tool:"database_lookup"`
- Real-time log streaming

**Config example:**

```yaml
# logstash.conf
input {
  tcp {
    port => 5000
    codec => json
  }
}

filter {
  # Parse agent logs
  if [event_type] == "tool_call" {
    mutate {
      add_field => { "[@metadata][index_name]" => "agent-tools" }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "%{[@metadata][index_name]}-%{+YYYY.MM.dd}"
  }
}
```

**Queries in Kibana:**
```
event_type: "error" AND agent_name: "support_agent"
  → See all errors from support agent

avg(latency_ms) by tool_name
  → Find slowest tools

error_rate: > 0.05 AND time: last_hour
  → Alert if error rate spikes
```

## Metrics: Prometheus

**What to export:**

```python
from prometheus_client import Counter, Histogram, Gauge

# Counter: how many requests processed
agent_requests_total = Counter(
    'agent_requests_total',
    'Total agent requests',
    ['agent_name', 'status']  # Can filter by agent and success/failure
)

# Histogram: latency distribution
agent_latency_seconds = Histogram(
    'agent_latency_seconds',
    'Agent request latency',
    ['agent_name'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0)  # P50, P95, P99 auto-calculated
)

# Gauge: current value
agent_active_requests = Gauge(
    'agent_active_requests',
    'Currently processing requests',
    ['agent_name']
)

# Token usage
tokens_used_total = Counter(
    'tokens_used_total',
    'Total tokens consumed',
    ['agent_name', 'model', 'type'],  # type: input, output
)

# Tool usage
tool_calls_total = Counter(
    'tool_calls_total',
    'Tool invocations',
    ['agent_name', 'tool_name', 'status']  # status: success, timeout, error
)

# Usage in code
agent_requests_total.labels(agent_name="support", status="success").inc()
agent_latency_seconds.labels(agent_name="support").observe(latency)
```

**Prometheus scrape config:**

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'agent-service'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

## Traces: Jaeger or Zipkin

Jaeger is better for NVIDIA ecosystems (used in NIM deployments). Shows distributed traces as interactive waterfall charts.

**Key metrics from traces:**
- P50, P95, P99 latency per service
- Slowest spans (where time is spent)
- Error spans (where failures occur)
- Service dependencies

---

# NVIDIA-Specific Monitoring

## NIM Health Endpoints

NVIDIA NIMs expose health check endpoints:

```bash
# Check if NIM is healthy
curl http://nim-server:8000/health

# Response (NIM should be ready before taking requests)
{
  "status": "healthy",
  "ready": true,
  "version": "1.1.0"
}

# Metrics endpoint (Prometheus format)
curl http://nim-server:8000/metrics

# Outputs:
# nim_request_duration_seconds{endpoint="/v1/completions"} 0.45
# nim_inference_latency_seconds 0.38
# nim_token_throughput_per_second 1200
# nim_queue_depth 5
```

## GPU Metrics: DCGM and nvidia-smi

Monitor GPU health since agent inference runs on GPUs.

```bash
# Real-time GPU status
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature --format=csv

# GPU 0: Tesla H100 | 95% util | 40GB/80GB | 65°C
# GPU 1: Tesla H100 | 87% util | 38GB/80GB | 62°C
```

**DCGM (Data Center GPU Manager) - Deeper metrics:**

```python
from pydcgm import dcgm_agent

# Query GPU health metrics
dcgm_agent.dcgmInit()
handle = dcgm_agent.dcgmConnect()

# Get power usage
power_usage = dcgm_agent.dcgmEngGetLatestValues(
    handle, gpu_id=0, field_id=dcgm_agent.DCGM_FI_DEV_POWER_USAGE
)

# Get thermal throttling events
thermal_throttle = dcgm_agent.dcgmEngGetLatestValues(
    handle, gpu_id=0, field_id=dcgm_agent.DCGM_FI_DEV_THERMAL_VIOLATIONS
)

# Get memory errors
memory_errors = dcgm_agent.dcgmEngGetLatestValues(
    handle, gpu_id=0, field_id=dcgm_agent.DCGM_FI_DEV_XID_ERRORS
)
```

**Metrics to monitor:**

| Metric | Alert Threshold | Meaning |
| --- | --- | --- |
| GPU Util | <20% | GPU idle, wasting resources |
| GPU Util | >95% | GPU throttled, may cause latency spikes |
| Memory Used | >90% | Risk of OOM |
| Temperature | >80°C | Thermal throttling imminent |
| Power Draw | >100% rated | Will throttle or power-cut |
| XID Errors | >0 | GPU encountered hardware error |

## Triton Inference Server Metrics

Triton exposes metrics about model performance:

```python
# Connect to Triton metrics
from prometheus_client.parser import TextFileToMetricFamilies
import requests

resp = requests.get("http://triton-server:8002/metrics")

# Parse metrics
for metric in TextFileToMetricFamilies(resp.text):
    print(metric.name, metric.documentation)
    # nv_inference_request_total: Total requests
    # nv_inference_exec_count: Execution count per model
    # nv_inference_queue_duration_us: Queue wait time
    # nv_inference_compute_input_duration_us: Input processing time
    # nv_inference_compute_infer_duration_us: Actual inference time
    # nv_inference_compute_output_duration_us: Output processing time
```

**Key Triton metrics:**

| Metric | Interpretation |
| --- | --- |
| `nv_inference_queue_duration_us` HIGH | Requests waiting in queue (batching issue) |
| `nv_inference_compute_infer_duration_us` HIGH | Inference itself is slow |
| `nv_inference_request_failure` HIGH | Model errors, version mismatch, bad input |
| `nv_inference_exec_count` by model_name | Which models being used |

---

# Real-Time Decision Tracking Dashboard

Build a Grafana dashboard showing live agent behavior.

## Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ Agent Monitoring Dashboard                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [Overall Metrics]                                              │
│  ┌─────────────┬──────────────┬─────────────┬──────────────┐   │
│  │ Req/min     │ Error Rate   │ Avg Latency │ P99 Latency  │   │
│  │ 1,250       │ 0.8%         │ 1,240ms     │ 4,100ms      │   │
│  └─────────────┴──────────────┴─────────────┴──────────────┘   │
│                                                                   │
│  [Tool Performance]          │  [Error Breakdown]                │
│  ├─ web_search: 450ms        │  ├─ timeout: 45%                │
│  ├─ db_lookup: 320ms         │  ├─ rate_limit: 30%             │
│  ├─ api_call: 280ms          │  └─ invalid_args: 25%           │
│  └─ cache_hit: 50ms          │                                   │
│                                                                   │
│  [Request Latency Distribution]                                 │
│  └─ Histogram: <100ms (5%) | 100-500ms (40%) | 500-1s (35%)   │
│                                                                   │
│  [Recent Requests (live)]                                       │
│  ├─ req_abc123: ✓ 1,200ms | 2 tools | "What's NVIDIA stock?" │
│  ├─ req_def456: ✗ 2,500ms | error | "Get latest earnings"     │
│  └─ req_ghi789: ⏳ 450ms [in progress...] | "Summarize news"  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Grafana Queries for Monitoring

```json
{
  "dashboard": {
    "title": "Agent Monitoring",
    "panels": [
      {
        "title": "Requests per minute",
        "targets": [
          {
            "expr": "rate(agent_requests_total[1m])"
          }
        ]
      },
      {
        "title": "P99 Latency by Agent",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, agent_latency_seconds_bucket)"
          }
        ]
      },
      {
        "title": "Tool Performance",
        "targets": [
          {
            "expr": "avg by (tool_name) (agent_tool_latency_seconds)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(agent_requests_total{status='error'}[5m]) / rate(agent_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Token Usage Rate",
        "targets": [
          {
            "expr": "rate(tokens_used_total[1m])"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "nvidia_smi_utilization_gpu"
          }
        ]
      }
    ]
  }
}
```

---

# Exam Practice Questions

**Q1:** You're debugging an agent that takes 3 seconds on average but occasionally takes 15 seconds. Your observability setup shows:
- Average latency: 3000ms
- P99 latency: 15000ms
- Tool call latency varies: web_search (400-2000ms), database (100-300ms)

What's the likely issue?

A) LLM inference is too slow
B) Variable network latency in web_search tool
C) Database query needs optimization
D) GPU is not at high utilization

**Answer: B** - The web_search tool shows huge variance (400-2000ms), which explains why P99 is 5x average. Database is consistent. This is a network/external service latency issue.

---

**Q2:** Your agent logs show structured JSON, but your Kibana queries are slow. What's the best optimization?

A) Add more Elasticsearch nodes
B) Add `trace_id` field as an Elasticsearch keyword (not analyzed text)
C) Switch to Grafana instead of Kibana
D) Reduce log volume by sampling

**Answer: B** - Keyword fields are indexed efficiently. Analyzing trace_ids as text is wasteful. This is a schema design issue, not a scale issue.

---

**Q3:** Your Jaeger traces show 90% of latency is in the "tool_execution" span. Which NVIDIA component should you check first?

A) NIM's `/health` endpoint
B) Triton's `nv_inference_queue_duration_us` metric
C) The actual tool (external API, database, etc) not NVIDIA components
D) GPU temperature from DCGM

**Answer: C** - If tool_execution is the bottleneck, that's the external service being called by the agent. While you should monitor NVIDIA components, this issue is outside NVIDIA. Actually verify the tool is the problem before assuming it's NVIDIA.

---

**Q4:** You want to track which agents are making the most tool calls. How should you instrument this?

A) Add a counter metric `tool_calls_total` with label `agent_name`
B) Log every tool call to Elasticsearch and aggregate in Kibana
C) Manually count from Jaeger traces
D) Query NIM's `/metrics` endpoint

**Answer: A** - Prometheus metrics with labels are designed for this. You can query `tool_calls_total` grouped by agent_name. Elasticsearch works too but metrics are more efficient for aggregation.

---

# Common Pitfalls

| Pitfall | Why It's Wrong | Fix |
| --- | --- | --- |
| "Log everything" | Observability systems crash under volume | Log at decision points only |
| "Just use print statements" | No structure, can't query or aggregate | Use structured JSON logging |
| "Tracing adds latency" | Can add 5-10% if misconfigured | Batch span processor, sample if needed |
| "Wait for error to debug" | By then, other requests completed | Trace every request, store traces for analysis |
| "Monitor NVIDIA components only" | Agent latency is often external tools | Trace the full end-to-end flow |
| "Ignore GPU metrics" | Thermal throttling silently degrades performance | Monitor temps, power, XID errors |

---

# Key Takeaways

1. **Structured logging** at decision boundaries, not everywhere
2. **Trace IDs** propagate through entire request flow (OpenTelemetry)
3. **ReAct traces** capture agent reasoning: thought → action → observation
4. **Observability stack:** ELK for logs, Prometheus for metrics, Jaeger for traces
5. **NVIDIA monitoring:** NIM health endpoints, GPU metrics (nvidia-smi, DCGM), Triton metrics
6. **Dashboards** show real-time agent behavior and decision chains
7. **Align observations with decisions:** If P99 >> avg latency, check tool variance not batching

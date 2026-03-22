# Alerting, Anomaly Detection, and Maintenance (8.3, 8.4, 8.5)

Running agents in production means knowing when things break, fixing them without downtime, and managing versions of models, prompts, and tools. This section covers alerting strategies, deployment patterns, and version management.

---

# Defining Alert Thresholds

Alerting is critical: you want to know about problems before users do. But false alarms are worse than missing a real problem (alert fatigue makes people ignore alerts).

## Key Agent Metrics to Alert On

| Metric | Warning Threshold | Critical Threshold | Root Cause |
| --- | --- | --- | --- |
| **Latency (P99)** | >3s (2x baseline) | >10s | Queueing, tool slowdown, or resource bottleneck |
| **Error Rate** | >2% | >5% | Model issues, tool failures, rate limits, bad input |
| **Tool Success Rate** | <95% | <90% | Tool API changes, auth issues, timeouts |
| **Token Cost per Request** | +50% from baseline | +100% | Prompt bloat, longer reasoning loops |
| **GPU Utilization** | <20% (idle) | N/A | Under-provisioned, wasting $ |
| **GPU Utilization** | >90% | >95% | Over-provisioned, risk of latency spikes |
| **Queue Depth** | >50 | >200 | Traffic spike or inference slowdown |
| **Model Accuracy** | -5% from baseline | -10% | Model drift, prompt changes, distribution shift |

## Alert Definition in Code (using Alertmanager)

```yaml
# alerting rules (prometheus)
groups:
  - name: agent_alerts
    rules:
      # Latency spike detection
      - alert: AgentLatencyHigh
        expr: histogram_quantile(0.99, agent_latency_seconds) > 10
        for: 5m
        annotations:
          summary: "Agent P99 latency > 10s"
          description: "P99 latency is {{ $value }}s (alert: >10s)"

      # Error rate spike
      - alert: AgentErrorRateHigh
        expr: |
          rate(agent_requests_total{status="error"}[5m]) /
          rate(agent_requests_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: "Agent error rate > 5%"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # Tool is failing
      - alert: ToolFailureRate
        expr: |
          rate(tool_calls_total{status="error"}[5m]) /
          rate(tool_calls_total[5m]) > 0.10
        for: 10m
        annotations:
          summary: "Tool {{ $labels.tool_name }} failing >10%"
          runbook: "Check tool logs: {{ $labels.tool_name }}"

      # Token cost exploding
      - alert: TokenCostSpike
        expr: |
          rate(tokens_used_total[5m]) >
          avg_over_time(rate(tokens_used_total[5m])[1h:5m]) * 1.5
        for: 10m
        annotations:
          summary: "Token usage spiked 50% above average"

      # GPU not being used
      - alert: GPUUnderutilized
        expr: avg(nvidia_smi_utilization_gpu) < 20
        for: 30m
        annotations:
          summary: "GPU utilization <20% for 30m"
          impact: "Wasting GPU resources"

      # GPU thermal issues
      - alert: GPUThermalThrottle
        expr: nvidia_smi_temperature > 80
        for: 5m
        annotations:
          summary: "GPU temperature >80°C"
          impact: "Performance degradation, possible shutdown"
```

## Alert Severity Levels

| Severity | Response Time | Action |
| --- | --- | --- |
| **Info** | No action | Just monitoring, log for later analysis |
| **Warning** | Within 1 hour | Team reviews, may need investigation |
| **Critical** | Within 5 minutes | Page on-call engineer immediately |
| **Emergency** | Immediate | Page all senior engineers |

**Example escalation:**
```
Error rate 2-5% → Warning (check logs)
Error rate 5-10% → Critical (page on-call)
Error rate >10% → Emergency (page team lead)
```

---

# Anomaly Detection: Beyond Static Thresholds

Static thresholds fail when patterns change. Anomaly detection learns baselines and detects deviations.

## Statistical Baseline Approach

Instead of "latency > 3 seconds", use "latency > 2 std deviations from mean".

```python
import numpy as np
from collections import deque

class AnomalyDetector:
    def __init__(self, window_size=300):  # 5-minute window
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)

    def add_measurement(self, latency_ms: float) -> dict:
        """Check if latency is anomalous."""
        self.latencies.append(latency_ms)

        if len(self.latencies) < 30:  # Need minimum samples
            return {"anomaly": False, "reason": "collecting_baseline"}

        mean = np.mean(self.latencies)
        std = np.std(self.latencies)
        zscore = (latency_ms - mean) / std

        is_anomaly = abs(zscore) > 2.5  # 2.5 std deviations = ~1% outliers

        return {
            "anomaly": is_anomaly,
            "zscore": zscore,
            "mean": mean,
            "std": std,
            "current": latency_ms,
            "expected_range": (mean - 2*std, mean + 2*std)
        }
```

## Query Distribution Drift Detection

Agents may behave differently with different query types. Detect when query distribution shifts.

```python
from collections import Counter

class QueryDriftDetector:
    def __init__(self, baseline_window=1000):
        self.baseline_window = baseline_window
        self.baseline_queries = deque(maxlen=baseline_window)
        self.current_queries = deque(maxlen=baseline_window)

    def categorize_query(self, query: str) -> str:
        """Rough categorization of query type."""
        if "price" in query.lower():
            return "pricing_query"
        elif "technical" in query.lower() or "how" in query.lower():
            return "technical_query"
        elif "news" in query.lower():
            return "news_query"
        else:
            return "general_query"

    def add_query(self, query: str, is_baseline=False):
        """Track query distribution."""
        category = self.categorize_query(query)

        if is_baseline:
            self.baseline_queries.append(category)
        else:
            self.current_queries.append(category)

    def detect_drift(self) -> dict:
        """Compare current distribution to baseline."""
        if len(self.baseline_queries) < 100:
            return {"drift": False, "reason": "collecting_baseline"}

        baseline_dist = Counter(self.baseline_queries)
        current_dist = Counter(self.current_queries)

        # Chi-square test for distribution change
        chi_square = 0
        for category in baseline_dist:
            expected = baseline_dist[category] / len(self.baseline_queries)
            observed = current_dist.get(category, 0) / len(self.current_queries)
            if expected > 0:
                chi_square += ((observed - expected) ** 2) / expected

        drift_detected = chi_square > 3.841  # Chi-square threshold for p=0.05

        return {
            "drift_detected": drift_detected,
            "chi_square": chi_square,
            "baseline_dist": dict(baseline_dist),
            "current_dist": dict(current_dist),
            "recommendation": "Consider retraining or updating prompts" if drift_detected else "OK"
        }
```

## Prometheus Anomaly Detection

Use PromLens or Grafana to automatically detect anomalies:

```yaml
# Prometheus anomaly detection using prediction
- alert: LatencyAnomalyDetected
  expr: |
    abs(agent_latency_seconds - predict_linear(agent_latency_seconds[1h], 3600))
    > 0.5
  for: 5m
  annotations:
    summary: "Latency deviated from predicted trend"
```

---

# Alert Routing and On-Call

## Alert Routing Rules

```yaml
# alertmanager config
global:
  resolve_timeout: 5m

route:
  receiver: team-default
  group_by: [alertname, agent_name]
  group_wait: 10s
  group_interval: 10s

  routes:
    # Critical alerts go to PagerDuty
    - match:
        severity: critical
      receiver: pagerduty-oncall
      continue: true

    # Tool failures go to Slack #tool-alerts
    - match:
        alertname: ToolFailureRate
      receiver: slack-tools

    # GPU alerts go to infrastructure team
    - match:
        alertname: GPUThermalThrottle
      receiver: slack-infra

receivers:
  - name: pagerduty-oncall
    pagerduty_configs:
      - service_key: ${PAGERDUTY_SERVICE_KEY}
        description: "{{ .GroupLabels.alertname }}"
        client: "Agent Monitoring"

  - name: slack-tools
    slack_configs:
      - api_url: ${SLACK_WEBHOOK_TOOLS}
        channel: "#tool-alerts"
        title: "Tool Failure Alert"
        text: "{{ .GroupLabels.tool_name }} failing at >10%"

  - name: slack-infra
    slack_configs:
      - api_url: ${SLACK_WEBHOOK_INFRA}
        channel: "#gpu-alerts"
        title: "GPU Alert"
```

## PagerDuty Integration Example

```python
# Automatic escalation
{
  "escalation_policy": {
    "escalation_rules": [
      {
        "delay_in_minutes": 0,
        "targets": [
          {"id": "oncall_engineer", "type": "schedule"}
        ]
      },
      {
        "delay_in_minutes": 15,
        "targets": [
          {"id": "senior_engineer", "type": "schedule"}
        ]
      },
      {
        "delay_in_minutes": 30,
        "targets": [
          {"id": "team_lead", "type": "user"}
        ]
      }
    ]
  }
}
```

---

# Blue-Green Deployment for Zero-Downtime Updates

Blue-green deployment runs two identical environments in parallel. You switch traffic between them instantly.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Load Balancer                                           │
│ (Routes 100% traffic to BLUE)                           │
└────────────┬────────────────────────────────────────────┘
             │
     ┌───────┴────────┐
     │                │
 ┌──▼────┐        ┌──▼────┐
 │ BLUE  │        │ GREEN  │
 │ (Live)│        │ (New)  │
 │       │        │        │
 │Model  │        │Model   │
 │v1.0   │        │v1.1    │
 │Prompt │        │Prompt  │
 │v2.3   │        │v2.4    │
 │Tools  │        │Tools   │
 │v3.0   │        │v3.0    │
 └──┬────┘        └───┬────┘
    │ 100% traffic    │ 0% traffic
    │                 │
    │ (Swap when ready)
    │
    └─ Now GREEN is 100%, BLUE is 0%
```

## Deployment Steps

1. **Deploy to GREEN (inactive environment)**
   - New model version
   - New prompt version
   - New tool versions
   - All healthchecks pass

2. **Smoke test GREEN**
   - Run automated tests
   - Try 10-20 test queries
   - Verify metrics look good

3. **Switch traffic instantly**
   - Load balancer redirects 100% to GREEN
   - Downtime: <100ms (DNS TTL)

4. **Monitor GREEN (now active)**
   - Watch error rates, latency, costs
   - Keep BLUE running for 1+ hour

5. **Rollback instantly if needed**
   - Switch back to BLUE (now inactive)
   - Downtime: <100ms

## Blue-Green Config (Kubernetes)

```yaml
# Blue deployment (currently active)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-blue
  labels:
    version: "1.0"
    deployment: blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent
      deployment: blue
  template:
    metadata:
      labels:
        app: agent
        deployment: blue
    spec:
      containers:
      - name: agent
        image: myregistry.azurecr.io/agent:v1.0
        env:
        - name: MODEL_VERSION
          value: "gpt-4-turbo-2025-02"
        - name: PROMPT_VERSION
          value: "2.3"
        - name: TOOLS_VERSION
          value: "3.0"
---
# Green deployment (new version, initially 0 replicas)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-green
  labels:
    version: "1.1"
    deployment: green
spec:
  replicas: 0  # Not running yet
  selector:
    matchLabels:
      app: agent
      deployment: green
  template:
    metadata:
      labels:
        app: agent
        deployment: green
    spec:
      containers:
      - name: agent
        image: myregistry.azurecr.io/agent:v1.1
        env:
        - name: MODEL_VERSION
          value: "gpt-4-turbo-2025-03"
        - name: PROMPT_VERSION
          value: "2.4"
        - name: TOOLS_VERSION
          value: "3.0"
---
# Service routes to whichever deployment is active
apiVersion: v1
kind: Service
metadata:
  name: agent-service
spec:
  selector:
    app: agent
    deployment: blue  # Currently routes to blue
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

## Switching Traffic (Blue to Green)

```bash
# Scale up GREEN
kubectl scale deployment agent-green --replicas=3

# Wait for GREEN to be healthy
kubectl rollout status deployment agent-green --timeout=5m

# Switch traffic: update service selector
kubectl patch service agent-service -p '{"spec":{"selector":{"deployment":"green"}}}'

# Monitor GREEN (now active)
# If bad, immediately revert:
kubectl patch service agent-service -p '{"spec":{"selector":{"deployment":"blue"}}}'

# Scale down BLUE after successful GREEN run
kubectl scale deployment agent-blue --replicas=0
```

---

# Canary Deployments for Gradual Rollout

Canary deploys send a small percentage of traffic to new version, gradually increasing if healthy.

## Canary Deployment Strategy

```
Phase 1 (5 min):  5% traffic to v1.1
Phase 2 (10 min): 25% traffic to v1.1
Phase 3 (15 min): 50% traffic to v1.1
Phase 4 (20 min): 100% traffic to v1.1

If error rate spikes, immediately revert to 0%
```

## Istio/Flagger Configuration

```yaml
# Canary resource (Flagger)
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: agent-canary
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent
  # Traffic split
  service:
    port: 8000
  # Canary config
  skipAnalysis: false
  progressDeadlineSeconds: 300
  service:
    port: 8000
    targetPort: 8000
  # Gradual rollout
  analysis:
    interval: 1m
    threshold: 10  # Max 10% error rate increase
    maxWeight: 50  # Max 50% traffic to canary
    stepWeight: 5  # Increase by 5% each interval
    metrics:
    - name: error_rate
      thresholdRange:
        max: 5
      interval: 1m
    - name: latency_p99
      thresholdRange:
        max: 5000  # 5 seconds
      interval: 1m
  # Webhook for validation
  skipAnalysis: false
```

---

# Model Versioning and Registry

An "agent version" isn't just the model. It's: **model + prompt + tools + configuration**.

## Version Tracking Strategy

```
Agent Version: v2.1.0
├─ Model: gpt-4-turbo-2025-03 (updated)
├─ Prompt: customer_support_v2.1 (updated)
├─ Tools: [
│    ├─ knowledge_base: v3.2 (unchanged)
│    ├─ crm_integration: v4.1 (updated)
│    └─ email_sender: v1.0 (unchanged)
│  ]
├─ Configuration: {
│    "temperature": 0.7,
│    "max_tokens": 2000
│  }
├─ Created: 2026-03-22T14:30:00Z
├─ Created_by: alice@company.com
├─ Changelog: "Updated to use new CRM API, improved customer context"
└─ Status: APPROVED_FOR_PRODUCTION
```

## Model Registry (MLflow / HuggingFace Model Hub)

```python
import mlflow

# Register agent version
with mlflow.start_run():
    # Log model
    mlflow.keras.log_model(
        model=gpt4_model,
        artifact_path="model"
    )

    # Log prompt (as artifact)
    with open("prompt.txt", "w") as f:
        f.write("You are a customer support agent...")
    mlflow.log_artifact("prompt.txt")

    # Log tool versions
    mlflow.log_dict({
        "knowledge_base": "v3.2",
        "crm_integration": "v4.1",
        "email_sender": "v1.0"
    }, artifact_file="tool_versions.json")

    # Log config
    mlflow.log_params({
        "temperature": 0.7,
        "max_tokens": 2000,
        "model_name": "gpt-4-turbo-2025-03"
    })

    # Log metrics (validation accuracy, latency, cost)
    mlflow.log_metrics({
        "validation_accuracy": 0.94,
        "avg_latency_ms": 1200,
        "cost_per_request": 0.003
    })

    # Tag for easy lookup
    mlflow.set_tag("stage", "production")
    mlflow.set_tag("agent", "customer_support")
    mlflow.set_tag("created_by", "alice@company.com")

# Load specific version later
logged_model = mlflow.pyfunc.load_model("models:/agent-customer-support/Production")
result = logged_model.predict(input_data)
```

## Triton Model Repository Versioning

Triton stores models with versions. You can have v1, v2, v3 of the same model.

```
model_repository/
├── customer_support_agent/
│   ├── 1/           ← v1 (old)
│   │   ├── model.bin
│   │   └── prompt.txt
│   ├── 2/           ← v2 (current production)
│   │   ├── model.bin
│   │   ├── prompt.txt
│   │   └── config.pbtxt
│   └── 3/           ← v3 (canary testing)
│       ├── model.bin
│       ├── prompt.txt
│       └── config.pbtxt
└── config.pbtxt (defines which version is default)
```

```protobuf
# model_repository/customer_support_agent/config.pbtxt
name: "customer_support_agent"
platform: "onnxruntime_onnx"

# Default to v2
default_model_filename: "model.bin"

# Version policy: always use latest
model_repository_agents {
  agents {
    gpu_device: 0
  }
}
```

---

# Rollback Procedures

When to rollback: error rate spikes, latency increases unexpectedly, accuracy drops, or cost explodes.

## Immediate Rollback (Blue-Green)

```bash
# Instant rollback command
kubectl patch service agent-service \
  -p '{"spec":{"selector":{"deployment":"blue"}}}'

# Downtime: <100ms
# Old version immediately serves traffic
```

## Gradual Rollback (Canary)

If you rolled out v1.1 and it's bad:

```bash
# Reverse the canary: send all traffic back to v1.0
flagger rollback agent-canary

# This gradually shifts traffic back: 100% v1.1 → 0% v1.1
```

## Version Rollback Decision Tree

```
Error rate > 10%?
    ↓ YES
    → Rollback immediately (don't wait)

Latency increased >50%?
    ↓ YES
    → Rollback immediately

Accuracy dropped >5%?
    ↓ YES
    → Rollback immediately

Cost per request increased >30%?
    ↓ YES
    → Investigate, then rollback if justified

Everything looks good for 1+ hours?
    ↓ YES
    → Accept new version, clean up old version
```

## Rollback Post-Mortem

```
What went wrong?
├─ Model regression: Retrain on more data
├─ Prompt regression: Review new prompt version
├─ Tool integration bug: Debug and fix
├─ Config error: Adjust batching/concurrency
└─ Downstream service: Not a code issue

Resolution:
├─ Create issue to fix
├─ Add test case to prevent
├─ Wait for fix before re-deploying
└─ Increase canary duration next time
```

---

# NVIDIA NIM Version Management

NVIDIA NIM (NVIDIA Inference Microservice) has its own versions. Manage them carefully.

```bash
# Check NIM version
curl http://nim-server:8000/health | jq '.version'
# Output: "1.1.0"

# List available NIM versions in Docker registry
aws ecr describe-images \
  --repository-name nvidian/nim \
  --registry-id 123456789 \
  --query 'imageDetails[*].imageTags' | jq '.[] | sort'
# Output: ["1.0.0", "1.1.0", "1.2.0"]
```

## Upgrading NIM Versions

NIM upgrades can have breaking changes. Use blue-green:

```yaml
# Current deployment: NIM 1.1.0
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-blue
spec:
  containers:
  - name: nim
    image: nvidian/nim:1.1.0
    env:
    - name: NIM_CUDA_VISIBLE_DEVICES
      value: "0"
---
# New deployment: NIM 1.2.0 (new features, bug fixes)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-green
spec:
  replicas: 0  # Scale to 0 until ready
  containers:
  - name: nim
    image: nvidian/nim:1.2.0
    env:
    - name: NIM_CUDA_VISIBLE_DEVICES
      value: "1"
```

## Checking Backward Compatibility

Before upgrading NIM:

```bash
# Test new NIM version on same requests
# NIM 1.1.0 output
curl -X POST http://nim-blue:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is NVIDIA?"}' \
  > output_v1.1.0.json

# NIM 1.2.0 output
curl -X POST http://nim-green:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is NVIDIA?"}' \
  > output_v1.2.0.json

# Compare outputs
diff <(jq '.choices[0].text' output_v1.1.0.json) \
     <(jq '.choices[0].text' output_v1.2.0.json)
```

---

# Zero-Downtime Update Checklist

Before updating anything in production:

```
PRE-DEPLOYMENT:
  ☐ Have rollback procedure documented
  ☐ Have on-call engineer available
  ☐ Have health checks defined
  ☐ Have monitoring dashboard open
  ☐ Have test suite passing
  ☐ Have load test passing
  ☐ Have smoke tests written

DURING-DEPLOYMENT:
  ☐ Deploy to inactive environment (blue-green)
  ☐ Run smoke tests in inactive environment
  ☐ Check health endpoints responding
  ☐ Check metrics in Prometheus
  ☐ Verify no error spikes
  ☐ Switch traffic (gradual or instant)

POST-DEPLOYMENT:
  ☐ Monitor error rate (15 minutes minimum)
  ☐ Monitor latency (15 minutes minimum)
  ☐ Monitor token cost (15 minutes minimum)
  ☐ Check GPU health (temp, util, power)
  ☐ Check NIM health endpoint
  ☐ Have plan to rollback if needed
  ☐ Keep old version running for 1+ hour
  ☐ Accept new version once stable
  ☐ Document any issues for post-mortem
```

---

# Exam Practice Questions

**Q1:** You deploy agent v1.1 with a new prompt. After 10 minutes, error rate jumps from 0.8% to 8%. What do you do?

A) Wait 5 more minutes to see if it stabilizes
B) Immediately switch traffic back to v1.0 using blue-green rollback
C) Increase the error rate alert threshold
D) Scale up more agents to handle the load

**Answer: B** - 8% error rate meets the "rollback immediately" threshold. Blue-green allows instant rollback (<100ms downtime). Don't wait and hope it stabilizes.

---

**Q2:** You want to gradually roll out agent v1.1. Which approach minimizes risk?

A) Deploy to all servers at once, monitor
B) Deploy to 10% of servers, then 25%, 50%, 100%
C) Deploy to production and blue-green rollback if bad
D) Wait for 100% test coverage before deploying

**Answer: B** - Canary deployment (gradual rollout with automatic rollback on error spike) is the safest. Tests catch obvious bugs but not all production issues.

---

**Q3:** You notice agent latency increased from 1.2s to 2.1s after deploying new NIM version 1.2.0. You want to investigate before rolling back. What should you check first?

A) Rollback immediately without investigating
B) Check NIM health endpoint and metrics
C) Revert the prompt to the old version
D) Increase GPU memory allocation

**Answer: B** - NIM version changes can affect inference latency. Check `/health` endpoint and `nv_inference_compute_infer_duration_us` metric to confirm NIM is the issue. If NIM is fine, then investigate prompts/config.

---

**Q4:** Your agent version registry shows:
```
v2.0.0: model=gpt-4, prompt=v3, tools={kb:v2, crm:v1}
v2.1.0: model=gpt-4, prompt=v3, tools={kb:v3, crm:v1}
v2.2.0: model=gpt-4, prompt=v4, tools={kb:v3, crm:v2}
```

Version v2.1.0 is in production. You need to deploy v2.2.0 but want minimal risk. What should you do?

A) Deploy directly since only prompt changed
B) Deploy to 0% canary traffic, gradually increase, monitor for prompt regression
C) Update just the prompt first, then tools
D) Revert to v2.0.0 and deploy from there

**Answer: B** - Multiple components changed (prompt v4, crm v2). Canary deployment with monitoring for accuracy changes is safest. You don't know if the new prompt/tools combination works well together.

---

**Q5:** You have two separate agent deployments:
- agent-realtime: needs latency <500ms
- agent-batch: processes overnight reports

Which deployment strategy is best for each?

A) Blue-green for both (same strategy)
B) Canary for both (same strategy)
C) Blue-green for realtime, canary for batch
D) Canary for realtime, blue-green for batch

**Answer: C** - Realtime needs <500ms latency, so instant switch matters (blue-green). Batch jobs don't need instant switch, so gradual canary is fine (and safer for testing).

---

# Common Pitfalls

| Pitfall | Why It's Wrong | Fix |
| --- | --- | --- |
| "High error rate, let's just wait" | Problem compounds, more users affected | Rollback immediately |
| "Monitor for hours before deploying" | Adds hours to deployment, misses issues | Deploy gradually (canary) |
| "Only version the model, not the prompt" | Prompt changes matter too | Version = model + prompt + tools |
| "Rollback takes 30 minutes" | Users suffer downtime | Use blue-green for <100ms rollback |
| "Scale up more agents to fix latency" | Doesn't help if inference is slow | Investigate root cause first |
| "Deploy new NIM without testing" | Breaking changes cause outages | Blue-green new NIM before switching |

---

# Key Takeaways

1. **Alert thresholds** tied to business impact: latency, error rate, accuracy, cost
2. **Anomaly detection** learns baselines, detects 2+ std dev deviations
3. **Alert routing:** Critical to PagerDuty, operational to Slack, for faster response
4. **Blue-green deploys** for instant rollback, near-zero downtime
5. **Canary deploys** for gradual rollout with automatic rollback on errors
6. **Version everything:** Model + prompt + tools + config = agent version
7. **Model registry** (MLflow) tracks versions, parameters, metrics for reproducibility
8. **Rollback procedures** documented before deployment, practiced regularly
9. **NVIDIA NIM versions** may have breaking changes, use blue-green for upgrades
10. **Zero-downtime updates** require planning: testing, monitoring, rollback readiness

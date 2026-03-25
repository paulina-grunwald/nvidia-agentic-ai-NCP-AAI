# Domain 08: Run, Monitor, and Maintain (7% exam weight)

Complete study materials for agent monitoring, maintenance, and deployment.

## Files in This Domain

### 1. observability-and-tracing.md (Objectives 8.1, 8.2)

**What to Monitor:** Agent decision-making in real-time

Topics covered:

- Structured logging at decision boundaries (what to log: inputs, outputs, tool calls, reasoning traces)
- Distributed tracing with OpenTelemetry (trace IDs across services)
- ReAct tracing (thought → action → observation chains)
- Observability stack: ELK (logs), Prometheus (metrics), Jaeger/Zipkin (traces)
- NVIDIA-specific monitoring: NIM health endpoints, GPU metrics (nvidia-smi, DCGM), Triton metrics
- Real-time decision tracking dashboards in Grafana
- 4 exam MCQs

### 2. alerting-and-maintenance.md (Objectives 8.3, 8.4, 8.5)

**How to Fix Issues and Deploy Safely**

Topics covered:

- Alert thresholds: latency spikes, error rates, accuracy drops, cost overruns
- Anomaly detection: statistical baselines, query drift detection
- Alert routing: PagerDuty, Slack
- Blue-green deployment for instant rollback (<100ms downtime)
- Canary deployments for gradual rollout with automatic rollback
- Model versioning: model + prompt + tools + config
- MLflow and Triton model registry
- Rollback procedures and decision trees
- NVIDIA NIM version management
- Zero-downtime update checklist
- 5 exam MCQs

### 3. throughput-vs-latency.md (Pre-existing reference)

Performance baseline metrics and Triton batching configuration.

---

## Quick Study Guide

**Objective 8.1: Monitoring**

- Know what to log: inputs, outputs, tool calls, latency, tokens, errors
- Understand distributed tracing (trace IDs, OpenTelemetry)
- Remember: trace_id propagates through all services

**Objective 8.2: Real-time Decision Tracking**

- ReAct agents show their reasoning: thought → action → observation
- Capture this chain in structured logs
- Use Grafana dashboards to show live agent behavior

**Objective 8.3: Alerting**

- Set thresholds: latency >3s warning, >10s critical
- Error rate >2% warning, >5% critical
- Use anomaly detection (2+ std deviations from baseline)
- Route critical to PagerDuty (immediate response)

**Objective 8.4: Maintenance**

- Blue-green: deploy to inactive environment, switch traffic instantly
- Canary: gradually increase traffic (5% → 25% → 50% → 100%)
- Both allow instant rollback if error rate spikes

**Objective 8.5: Versioning**

- Agent version = model + prompt + tools + config
- Use MLflow or Triton registry to track versions
- Keep old version running for 1+ hour after deploy
- Practice rollback procedures regularly

---

## Exam Tips

- **P99 >> Average latency?** → Queue buildup (batching issue), not tool slowness
- **Error rate spikes?** → Rollback immediately, don't wait
- **Latency increase after deployment?** → Check NIM health and metrics first
- **Gradual vs instant rollout?** → Blue-green for realtime (<500ms SLA), canary for batch
- **GPU underutilized?** → Increase batch size (if latency allows) or accept wasted resources
- **Alert fatigue?** → Use anomaly detection instead of static thresholds

---

## Study Path

1. Read throughput-vs-latency.md (baseline understanding)
2. Read observability-and-tracing.md (understand your agent)
3. Read alerting-and-maintenance.md (fix issues, deploy safely)
4. Do all 9 exam MCQs
5. Create your own example: design monitoring for a support agent

---

## Lab Exercise Ideas

1. Set up ELK stack logging for an agent, write structured logs
2. Add OpenTelemetry to agent code, view traces in Jaeger
3. Define Prometheus alerts, test with canary
4. Deploy blue-green agent update, practice rollback
5. Create MLflow model registry entry for agent version
6. Write anomaly detection script that learns latency baseline

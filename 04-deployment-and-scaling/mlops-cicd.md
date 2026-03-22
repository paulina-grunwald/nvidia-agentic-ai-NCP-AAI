# MLOps & CI/CD for Agent Systems

## Overview

MLOps for agentic AI is different from traditional model serving. You're not just versioning a static model anymore. You're managing:

- **Prompt versions** - Agent instructions change
- **Tool definitions** - Tools agents can call evolve
- **Model versions** - Underlying LLMs get upgraded
- **Agent logic** - How agents reason and make decisions
- **Configuration** - Temperature, max tokens, retry policies, etc.

This section covers the full MLOps pipeline: from development through production governance.

---

## The Agent CI/CD Pipeline

### Traditional Model Pipeline (vs Agent Pipeline)

```
Traditional ML:
Code → Train → Validate → Push Model → Deploy → Serve

Agent Pipeline (More Complex):
Code → Version Prompts → Version Tools → Integration Test
→ Test Agent Reasoning → A/B Test Agents → Canary Deploy
→ Monitor Agent Quality → Rollback if needed
```

An agent change might be:
- A new tool the agent can use
- A system prompt refinement
- A model upgrade (Llama 7B → Llama 70B)
- A configuration change (temperature 0.7 → 0.5)

Each requires different validation before production.

---

## 1. Artifact Versioning

### What Gets Versioned?

| Artifact | Example | Why Version It |
|----------|---------|---|
| **Prompt** | System prompt for agent | Changes impact agent behavior fundamentally |
| **Tools** | Tool schema, function signatures | Tool changes affect what agent can do |
| **Model** | NIM version, Llama variant | Model version = inference characteristics |
| **Config** | Temperature, max_tokens, retry_count | Settings change inference behavior |
| **Dependencies** | langchain, anthropic SDK versions | Breaking changes impact compatibility |

### Semantic Versioning for Agents

Use **MAJOR.MINOR.PATCH**:

- **MAJOR** - Breaking change (tool removed, prompt incompatible with old tools, model downgrade)
- **MINOR** - Backward compatible enhancement (new tool added, better prompt, improved config)
- **PATCH** - Bug fix (typo in prompt, config value adjustment)

Example version progression:

```
v1.0.0 - Initial agent with 5 tools, Llama 8B
v1.1.0 - Added new tool (web search), still Llama 8B
v1.1.1 - Fixed typo in system prompt
v1.2.0 - Added caching tool, improved error handling
v2.0.0 - Upgraded to Llama 70B (breaking - different performance characteristics)
```

### Storing Artifacts in Version Control

**Git repository structure:**

```
agent-repo/
├── agents/
│   ├── research-agent/
│   │   ├── system_prompt.txt       # Agent instructions
│   │   ├── tools.yaml              # Tool definitions
│   │   ├── config.yaml             # Temperature, max tokens, etc.
│   │   └── VERSION                 # Semantic version (e.g., "v1.2.3")
│   └── coding-agent/
│       ├── system_prompt.txt
│       ├── tools.yaml
│       └── config.yaml
├── tests/
│   ├── test_research_agent.py
│   └── test_coding_agent.py
├── .github/workflows/
│   ├── test.yml
│   └── deploy.yml
└── registry/
    └── artifacts.yaml             # Manifest of versions
```

**artifacts.yaml** tracks all versions:

```yaml
agents:
  research-agent:
    current_version: v1.2.3
    available_versions:
      - v1.2.3
      - v1.2.2
      - v1.1.0
    model: nvcr.io/nim/meta/llama3-8b-instruct:latest

  coding-agent:
    current_version: v2.0.0
    available_versions:
      - v2.0.0
    model: nvcr.io/nim/meta/llama3-70b-instruct:latest
```

---

## 2. Model Registry & NIM Versioning

### NVIDIA NIM in MLOps Context

NIM (NVIDIA Inference Microservices) containers are versioned. When you deploy an agent using a NIM model, you need to:

1. Pin the NIM container version
2. Track which NIM version was used for which agent version
3. Be able to rollback to previous NIM versions if needed

### NIM Image Versioning

```
nvcr.io/nim/meta/llama3-70b-instruct:1.0.0
nvcr.io/nim/meta/llama3-70b-instruct:latest
```

**NVIDIA's versioning scheme:**

| Version | Meaning | Use Case |
|---------|---------|----------|
| `latest` | Most recent build | Development only (unstable) |
| `1.0.0` | Specific release | Production (stable) |
| `1.0` | Latest patch of 1.0.x | Staging (get updates) |

### Model Registry Pattern

Store NIM metadata in your registry:

```yaml
# models/registry.yaml
models:
  llama3-8b:
    source: nvcr.io/nim/meta/llama3-8b-instruct
    versions:
      1.0.0:
        date_released: 2024-09-15
        tested_with_agents:
          - research-agent:v1.0.0
          - coding-agent:v1.0.0
        known_issues: []
      1.0.1:
        date_released: 2024-09-20
        tested_with_agents:
          - research-agent:v1.1.0
        known_issues: ["occasional timeout on long contexts"]

  llama3-70b:
    source: nvcr.io/nim/meta/llama3-70b-instruct
    versions:
      1.0.0:
        date_released: 2024-10-01
        tested_with_agents:
          - research-agent:v2.0.0
        known_issues: []
```

### Docker Multi-Stage Build for Agents

When containerizing agents, include model pinning:

```dockerfile
# Dockerfile for agent with pinned NIM
FROM python:3.11-slim

# Pin dependencies including model client versions
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy agent artifacts
COPY agents/research-agent/ /app/agent/

# Set environment variables to pin NIM version
ENV NIM_MODEL_VERSION=1.0.0
ENV NIM_ENDPOINT=http://nim-service:8000

# Metadata - this gets into image labels
LABEL agent.version="1.2.3"
LABEL nim.model="llama3-70b"
LABEL nim.version="1.0.0"

WORKDIR /app
CMD ["python", "-m", "agent_service"]
```

---

## 3. Experiment Tracking & Evaluation

### Agent Quality Metrics

Unlike static models, agents need:

| Metric | What It Measures | How to Track |
|--------|---|---|
| **Task Success Rate** | % of tasks agent completes correctly | Manual annotation, golden dataset |
| **Tool Usage Correctness** | % of time agent calls right tool | Test against labeled examples |
| **Response Quality** | Is response helpful/accurate? | LLM-as-judge, human review |
| **Latency** | Time from request to response | Instrument agent code |
| **Token Efficiency** | Tokens used per task | Count tokens in agent reasoning |
| **Error Recovery** | Can agent recover from tool failures? | Chaos testing, synthetic failures |

### Tracking Experiments

Use a tool like **MLflow**, **Weights & Biases**, or **Neptune**:

```python
import mlflow
from datetime import datetime

def evaluate_agent_version(agent_version: str, test_dataset: str):
    """Evaluate agent version on test dataset"""

    mlflow.set_experiment(f"agent-evaluation")
    with mlflow.start_run(run_name=f"eval-{agent_version}"):

        agent = load_agent(agent_version)
        results = []

        for question, expected_answer in load_test_dataset(test_dataset):
            response = agent.query(question)

            # Evaluate with multiple metrics
            is_correct = check_correctness(response, expected_answer)
            latency = measure_latency(response)
            tokens_used = count_tokens(response)

            results.append({
                "correct": is_correct,
                "latency": latency,
                "tokens": tokens_used
            })

        # Aggregate metrics
        success_rate = sum(r["correct"] for r in results) / len(results)
        avg_latency = sum(r["latency"] for r in results) / len(results)
        avg_tokens = sum(r["tokens"] for r in results) / len(results)

        # Log to MLflow
        mlflow.log_metric("success_rate", success_rate)
        mlflow.log_metric("avg_latency_ms", avg_latency)
        mlflow.log_metric("avg_tokens_used", avg_tokens)
        mlflow.log_param("agent_version", agent_version)
        mlflow.log_param("test_dataset", test_dataset)

        return {
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "avg_tokens": avg_tokens
        }

# Run before promotion to production
results = evaluate_agent_version("v1.2.3", "test_set_2024")
if results["success_rate"] >= 0.95:
    print("✓ Ready for production")
else:
    print("✗ Below threshold, needs iteration")
```

### Comparison Dashboard

Track agent versions side-by-side:

```
Agent Version Comparison (on test_set_2024):

Version    Success Rate  Avg Latency  Tokens Used  Status
v1.0.0          92%        245ms       1200       Baseline
v1.1.0          93%        280ms       1250       Slight improvement
v1.2.0          95%        250ms       1180       ✓ Ready for prod
v2.0.0          94%        200ms       900        Better latency, verify quality
```

---

## 4. CI/CD Workflow

### GitHub Actions Pipeline Example

```yaml
# .github/workflows/agent-ci-cd.yml
name: Agent CI/CD Pipeline

on:
  push:
    branches: [main, staging]
    paths:
      - 'agents/**'
      - 'tests/**'
  pull_request:
    branches: [main]

jobs:
  # Stage 1: Syntax & Format Checks
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Check prompt syntax
        run: python scripts/validate_prompts.py

      - name: Validate tool definitions
        run: python scripts/validate_tools.py

      - name: Check version tags match
        run: python scripts/check_versions.py

  # Stage 2: Unit Tests
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run agent unit tests
        run: pytest tests/ -v

      - name: Test agent reasoning (mock)
        run: pytest tests/test_agent_reasoning.py

  # Stage 3: Integration Testing (Real LLM)
  integration-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up NIM service
        run: docker-compose up -d  # Start mock NIM

      - name: Integration test with actual model
        env:
          NIM_ENDPOINT: http://localhost:8000
        run: |
          python tests/integration/test_agent_with_nim.py

      - name: Evaluate on golden dataset
        run: python scripts/evaluate_agent.py --version ${{ github.sha }}

  # Stage 4: Build & Push Container
  build:
    needs: [lint, test, integration-test]
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Build agent container
        run: |
          AGENT_VERSION=$(cat agents/research-agent/VERSION)
          docker build -t myregistry/agents:$AGENT_VERSION .

      - name: Push to registry
        run: |
          docker push myregistry/agents:$AGENT_VERSION

  # Stage 5: Deploy Decision
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to staging first
        run: |
          AGENT_VERSION=$(cat agents/research-agent/VERSION)
          kubectl set image deployment/research-agent-staging \
            research-agent=myregistry/agents:$AGENT_VERSION

      - name: Run smoke tests on staging
        run: |
          python tests/smoke_tests.py --endpoint staging.agents.internal

      - name: Await approval for production
        run: |
          echo "Deployment to production requires manual approval"
          echo "Check PR for approval"

  # Stage 6: Production Deployment (Manual Approval)
  deploy-prod:
    needs: deploy
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request' && contains(github.event.pull_request.labels.*.name, 'approved-for-prod')
    steps:
      - uses: actions/checkout@v3

      - name: Canary deploy to 5% of traffic
        run: |
          AGENT_VERSION=$(cat agents/research-agent/VERSION)
          kubectl apply -f k8s/canary-deployment.yaml \
            --set image=myregistry/agents:$AGENT_VERSION \
            --set traffic_percentage=5

      - name: Monitor canary (5 min)
        run: |
          sleep 300
          python scripts/check_canary_health.py

      - name: Promote to 100% traffic
        run: |
          kubectl apply -f k8s/production-deployment.yaml
```

### Approval Gate Pattern

```yaml
# k8s/approval-gate.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-approval-log
data:
  approvals: |
    Version: v1.2.3
    Approved by: alice@company.com
    Timestamp: 2024-03-15T14:30:00Z
    Tests passed: 95% success rate
    Regression check: ✓ No regressions vs v1.1.0
    Ready for production: YES
```

---

## 5. Deployment Strategies

### A/B Testing Agents

Compare two agent versions with live traffic:

```python
# Agent router for A/B testing
from random import random

class AgentRouter:
    def __init__(self, version_a: str, version_b: str, split_ratio: float = 0.5):
        self.agent_a = load_agent(version_a)
        self.agent_b = load_agent(version_b)
        self.split_ratio = split_ratio  # % of traffic to version A

    def route_request(self, request: str) -> dict:
        """Route request to A or B based on split"""
        if random() < self.split_ratio:
            version_used = "A"
            agent = self.agent_a
        else:
            version_used = "B"
            agent = self.agent_b

        response = agent.query(request)

        # Log for analysis
        log_metric(
            version=version_used,
            success=response.get("success"),
            latency=response.get("latency"),
            timestamp=datetime.now()
        )

        return {
            "response": response["text"],
            "version_used": version_used,
            "metadata": {
                "tokens_used": response.get("tokens"),
                "tools_called": response.get("tools")
            }
        }

# Use case: Test new prompt in production
router = AgentRouter("v1.1.0", "v1.2.0", split_ratio=0.5)
# Send 50% traffic to each version
# Measure success rate, latency, user satisfaction
# If v1.2.0 wins, promote it
```

### Canary Deployment

Roll out new agent version gradually:

```yaml
# k8s/canary-agent-deployment.yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: research-agent
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: research-agent

  # Gradually increase traffic to new version
  progressDeadlineSeconds: 300
  service:
    port: 8000

  analysis:
    interval: 1m
    threshold: 5  # Max 5 failed checks before rollback
    metrics:
    - name: error_rate
      thresholdRange:
        max: 0.05  # Allow up to 5% error rate
    - name: latency
      thresholdRange:
        max: 300   # Max 300ms latency

    webhooks:
    - name: acceptance-test
      url: http://flagger-loadtester/
      timeout: 30s
      metadata:
        type: bash
        cmd: "curl http://research-agent:8000/health"

  # Traffic split schedule
  skipAnalysis: false
  maxWeight: 100      # Max % traffic to new version
  stepWeight: 10      # Increase by 10% every minute
```

### Blue-Green Deployment

Keep both versions running, switch traffic instantly:

```bash
# Deployment v1 (Blue - Current)
kubectl apply -f k8s/research-agent-blue.yaml

# Deployment v2 (Green - New)
kubectl apply -f k8s/research-agent-green.yaml

# Service initially routes to Blue
kubectl apply -f k8s/service-blue.yaml

# ... run tests on Green ...

# If tests pass, switch service to Green
kubectl patch service research-agent-service \
  -p '{"spec":{"selector":{"version":"green"}}}'

# Keep Blue running for instant rollback
```

---

## 6. Governance & Rollback

### Production Checklist

Before promoting any agent version to production:

```markdown
# Agent Version v1.2.3 Promotion Checklist

## Code Review
- [ ] Code reviewed and approved by 2 engineers
- [ ] No security issues found
- [ ] Prompt changes explained in PR

## Testing
- [ ] Unit tests pass (100%)
- [ ] Integration tests pass
- [ ] Latency < 300ms (p95)
- [ ] Success rate >= 95% on golden dataset

## Compliance
- [ ] Model version supports all required tools
- [ ] Audit logging enabled
- [ ] Data retention policies met
- [ ] PII handling verified

## Monitoring
- [ ] Dashboards created
- [ ] Alerts configured
- [ ] Rollback procedure documented
- [ ] On-call owner assigned

## Approval
- [ ] Technical lead: ___________
- [ ] Product owner: ___________
- [ ] Security team: ___________
- [ ] Deployment timestamp: ___________
```

### Rollback Procedure

Agent deployments must be easily reversible:

```bash
#!/bin/bash
# scripts/rollback-agent.sh

AGENT_NAME=$1
ROLLBACK_VERSION=$2

echo "Rolling back $AGENT_NAME to version $ROLLBACK_VERSION"

# Update to previous version
kubectl set image deployment/$AGENT_NAME \
  $AGENT_NAME=myregistry/agents:$ROLLBACK_VERSION

# Wait for rollout
kubectl rollout status deployment/$AGENT_NAME

# Verify health
if [ $(curl http://$AGENT_NAME/health) == "ok" ]; then
  echo "✓ Rollback successful"

  # Log rollback event
  kubectl logs deployment/$AGENT_NAME | tail -20
else
  echo "✗ Health check failed, investigating..."
  kubectl describe deployment/$AGENT_NAME
fi
```

### Audit Trail

Track every agent change in production:

```python
# Audit logging for agent changes
class AuditLog:
    def log_deployment(self, agent_name: str, version: str, user: str, reason: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "version": version,
            "deployed_by": user,
            "reason": reason,
            "status": "deployed"
        }
        # Store in immutable log (database, cloud logging, etc.)
        self.storage.append(entry)

    def log_rollback(self, agent_name: str, from_version: str, to_version: str, reason: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "rolled_back_from": from_version,
            "rolled_back_to": to_version,
            "reason": reason,
            "status": "rolled_back"
        }
        self.storage.append(entry)

# Example usage
audit = AuditLog()
audit.log_deployment("research-agent", "v1.2.3", "alice@company.com",
                     "Improved prompt, 95% success rate")
```

---

## 7. NVIDIA-Specific Practices

### Using NVIDIA Catalog Models

When deploying NVIDIA's pre-trained models:

```yaml
# agents/research-agent/model-config.yaml
model:
  source: nvcr.io/nvidia/tao  # NVIDIA Catalog
  name: meta/llama3-70b-instruct
  version: "1.0"

  # Pin exact version in production
  image: nvcr.io/nim/meta/llama3-70b-instruct:1.0.0

  # Configuration
  config:
    max_batch_size: 32
    max_context_length: 4096
    precision: float16

  # NVIDIA NIM-specific settings
  nim:
    enable_kv_cache_reuse: true
    enable_paged_attention: true
    enable_inflight_batching: true
```

### Model Repository Manifest

When using Triton + NIM:

```yaml
# triton/model_repository/manifest.yaml
models:
  - name: research-agent
    version: 1  # Triton version
    backend: python

    config:
      agent_version: "v1.2.3"
      prompt_hash: "abc123def456"
      tools_version: "v2.1.0"

      # Triton batching
      dynamic_batching:
        preferred_batch_size: [8, 16, 32]
        max_queue_delay_microseconds: 1000
```

---

## Exam-Focused Summary

**Key patterns you'll see:**

| Scenario | Answer |
|----------|--------|
| "Need to track which NIM version was used with agent v1.2.3" | Model registry with version mapping |
| "Agent prompt changed, how to validate before production?" | Experiment tracking on golden dataset, approval gates |
| "Deploy new prompt version safely to 1M users" | Canary deployment or A/B testing |
| "Agent caused production issue, need to revert immediately" | Blue-green or rollback script |
| "Prove agent v2 is better than v1 for compliance audit" | MLflow experiment log with metrics |
| "Update to new Llama 70B NIM version, ensure compatibility" | Pin version in Docker, test in staging |

**Key distinctions:**
- Agents need **behavioral versioning** (prompt, tools, config), not just model versioning
- NVIDIA NIM versions matter for production stability
- Experiment tracking is mandatory for agent quality gates
- Governance requires approval gates and audit trails

---

## Exam Practice Questions

**Question 1:** You need to deploy agent version v1.2.3 that uses Llama 70B NIM version 1.0.1 to production. What should you do first?

A) Push the NIM image to registry and immediately deploy
B) Create a test in a staging environment with the exact NIM version, evaluate agent success rate, get approval
C) Deploy to 10% of traffic using canary deployment
D) Run unit tests only, then rollback plan

**Correct:** B - You need to verify the agent works with that specific NIM version in a controlled environment before production.

---

**Question 2:** Agent v1.1.0 in production is causing 2% failures. Version v1.0.0 had 0.5% failure rate. What should happen?

A) Immediately rollback to v1.0.0 using blue-green deployment
B) Contact the NIM team about the issue
C) A/B test v1.1.0 and v1.0.0, measure differences, then decide
D) Update the approval checklist and try again

**Correct:** A - This is a production incident. Rollback first to restore service, then investigate why v1.1.0 failed.

---

**Question 3:** How should you version the system prompt of an agent?

A) Don't version it, just update it in place
B) Use semantic versioning tied to the agent version (agent v1.2.3 implies prompt v1.2.3)
C) Version it separately and track prompt hash in artifact registry
D) Version it in git but don't track it in deployment manifests

**Correct:** B/C - The prompt is core to agent behavior and should be tracked. Either tie it to agent versioning or maintain a separate prompt manifest with hashes.

---

## Key Takeaways

1. **MLOps for agents is behavioral versioning** - track prompts, tools, and configs, not just models
2. **Experiment tracking is mandatory** - before any production deployment, evaluate on golden dataset
3. **Approval gates protect production** - require sign-offs from multiple teams
4. **Rollback must be instant** - keep previous version running or have scripts ready
5. **NIM versioning is critical** - pin exact NIM container versions in production deployments
6. **Governance requires audit trails** - log every deployment, rollback, and approval for compliance

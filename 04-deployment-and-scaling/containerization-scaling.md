# Containerization & Scaling for Agent Deployments

## Overview

Containerizing agentic AI systems is different from containerizing traditional services. Agents are stateful (they maintain context), may need persistent storage (for tool outputs, logs), and require careful resource allocation because they interact with expensive LLM services.

This guide covers:
- Dockerizing agents and their dependencies (NIM, Triton, tools)
- Kubernetes deployment patterns for stateful agents
- Load balancing strategies that understand GPU constraints
- Cost optimization for agent infrastructure
- High availability patterns

---

## 1. Docker for Agent Systems

### What Gets Containerized?

```
Agent Container Stack:

┌─────────────────────────────────┐
│   Your Agent Code               │
│   (Python/JavaScript/etc)       │
├─────────────────────────────────┤
│   Agent Framework               │
│   (LangChain, Anthropic SDK,    │
│    LlamaIndex, etc)             │
├─────────────────────────────────┤
│   LLM Client Libraries          │
│   (openai, anthropic, etc)      │
├─────────────────────────────────┤
│   Base OS + Runtime             │
│   (Python 3.11, system libs)    │
└─────────────────────────────────┘

   + External Services (via network):
   ├─ NIM (NVIDIA Inference Microservices)
   ├─ Triton Inference Server
   ├─ Tool services (web search, DB, etc)
   ├─ Vector DB (for RAG)
   └─ Cache/KV store
```

Agents rarely run the LLM directly in the container. Instead, they call it over HTTP via NIM or a cloud API.

### Multi-Stage Dockerfile Pattern

```dockerfile
# Dockerfile for agent with optimized layers

# Stage 1: Builder - Compile dependencies
FROM python:3.11-slim as builder

WORKDIR /build

# Install build tools only in builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels
COPY requirements.txt .
RUN pip install --user --no-cache-dir --compile -r requirements.txt

# Stage 2: Runtime - Minimal dependencies
FROM python:3.11-slim

WORKDIR /app

# Copy only compiled wheels from builder (much smaller)
COPY --from=builder /root/.local /root/.local

# Copy agent code and configuration
COPY agents/ ./agents/
COPY config.yaml .
COPY tools/ ./tools/

# Set Python path to use precompiled wheels
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app:$PYTHONPATH

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Metadata for orchestration
LABEL agent.name="research-agent" \
      agent.version="v1.2.3" \
      agent.framework="langchain" \
      nim.model="llama3-70b" \
      nim.version="1.0.0"

# Non-root user for security
RUN useradd -m agent
USER agent

EXPOSE 8000

CMD ["python", "-m", "agent_service.main"]
```

### Environment Variables for NIM/LLM Configuration

```dockerfile
# Dockerfile snippet - environment setup

ENV NIM_ENDPOINT="http://nim-service:8000" \
    NIM_API_KEY="" \
    TRITON_ENDPOINT="http://triton-service:8000" \
    MODEL_NAME="llama3-70b-instruct" \
    MODEL_VERSION="1.0.0" \
    LOG_LEVEL="INFO" \
    MAX_TOKENS="2000" \
    TEMPERATURE="0.7" \
    RETRY_POLICY="exponential" \
    CACHE_TYPE="redis"
```

Then pass these at runtime:

```bash
docker run \
  -e NIM_ENDPOINT="http://nim-host:8000" \
  -e NIM_API_KEY="$NIM_KEY" \
  -e TEMPERATURE="0.5" \
  my-agent:v1.2.3
```

### Docker Compose for Local Development

```yaml
# docker-compose.yml - local dev environment

version: '3.8'

services:
  # NVIDIA NIM Service (LLM inference)
  nim:
    image: nvcr.io/nim/meta/llama3-8b-instruct:1.0.0
    container_name: nim-service
    ports:
      - "8000:8000"
    environment:
      NIM_HTTP_PORT: 8000
      NIM_GRPC_PORT: 8001
    shm_size: 8gb  # Shared memory for GPU operations
    ipc: host      # Required for GPU memory sharing
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Agent service
  agent:
    build: .
    container_name: agent-service
    ports:
      - "5000:8000"  # Agent API on 5000, expects NIM on 8000
    environment:
      NIM_ENDPOINT: "http://nim:8000"
      LOG_LEVEL: DEBUG
    depends_on:
      - nim
    volumes:
      - ./agents:/app/agents:ro
      - ./logs:/app/logs
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          memory: 4G

  # Vector database for RAG
  milvus:
    image: milvusdb/milvus:v0.4.0
    container_name: milvus-service
    ports:
      - "19530:19530"
      - "9091:9091"

  # Redis for caching
  redis:
    image: redis:7-alpine
    container_name: redis-service
    ports:
      - "6379:6379"

volumes:
  milvus_data:
  redis_data:
```

Usage:

```bash
# Start entire stack
docker-compose up -d

# Check agent is responding
curl http://localhost:5000/health

# View agent logs
docker-compose logs -f agent

# Stop everything
docker-compose down
```

### Container Best Practices for Agents

| Practice | Why for Agents |
|----------|---|
| **Non-root user** | Security - agent shouldn't run as root |
| **Health checks** | Agents may hang waiting for external service |
| **Shared memory (`shm_size`)** | If agent uses GPU operations (embeddings, etc) |
| **Volume mounts for logs** | Audit trail of agent actions required |
| **Resource limits** | Prevent runaway agents from consuming all memory |
| **Specific base image versions** | Avoid surprises with Python/lib updates |

---

## 2. Kubernetes Deployments for Agents

### Basic Deployment Pattern

```yaml
# k8s/agent-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: research-agent
  labels:
    app: research-agent
    version: v1.2.3
spec:
  replicas: 3  # Start with 3 instances

  selector:
    matchLabels:
      app: research-agent

  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # Allow 1 extra pod during update
      maxUnavailable: 0  # Never take down all pods

  template:
    metadata:
      labels:
        app: research-agent
        version: v1.2.3
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"

    spec:
      # Service account for RBAC
      serviceAccountName: research-agent

      # Init container to wait for dependencies
      initContainers:
      - name: wait-for-nim
        image: busybox:1.35
        command: ['sh', '-c', 'until nc -z nim-service 8000; do echo waiting for NIM; sleep 2; done']

      containers:
      - name: agent
        image: myregistry/agents:v1.2.3
        imagePullPolicy: IfNotPresent

        # Port exposure
        ports:
        - name: http
          containerPort: 8000
          protocol: TCP

        # Environment variables
        env:
        - name: NIM_ENDPOINT
          value: "http://nim-service:8000"
        - name: REDIS_HOST
          value: "redis-service"
        - name: LOG_LEVEL
          value: "INFO"
        - name: MAX_RETRIES
          value: "3"

        # Secrets for API keys
        envFrom:
        - secretRef:
            name: agent-secrets

        # Resource requests and limits
        resources:
          requests:
            cpu: "1"           # Min CPU guaranteed
            memory: "2Gi"      # Min memory guaranteed
            ephemeral-storage: "1Gi"  # Temp storage for logs
          limits:
            cpu: "4"           # Max CPU allowed
            memory: "8Gi"      # Max memory allowed

        # Liveness probe - restart if dead
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          failureThreshold: 3

        # Readiness probe - remove from load balancer if not ready
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          failureThreshold: 2

        # Volume mounts
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: logs
          mountPath: /app/logs
        - name: cache
          mountPath: /tmp/agent-cache

      # Pod-level configuration
      restartPolicy: Always
      terminationGracePeriodSeconds: 30  # Time to gracefully shutdown

      # Tolerations for GPU nodes
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule

      # Pod affinity - prefer same node for stateful agents
      affinity:
        podAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - research-agent
              topologyKey: kubernetes.io/hostname

      volumes:
      - name: config
        configMap:
          name: agent-config
      - name: logs
        emptyDir:
          sizeLimit: 5Gi
      - name: cache
        emptyDir:
          sizeLimit: 2Gi
```

### Service Exposing the Agent

```yaml
# k8s/agent-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: research-agent-service
spec:
  type: LoadBalancer  # For external traffic

  selector:
    app: research-agent

  ports:
  - name: http
    port: 80           # External port
    targetPort: 8000   # Pod port
    protocol: TCP

  # Session affinity for stateful agents
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800  # 3 hours
```

### StatefulSet for Agents with Persistent State

Some agents need to persist data between runs. Use StatefulSet instead of Deployment:

```yaml
# k8s/agent-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: memory-agent  # Agent with persistent memory/context
spec:
  serviceName: memory-agent-service
  replicas: 2

  selector:
    matchLabels:
      app: memory-agent

  template:
    metadata:
      labels:
        app: memory-agent
    spec:
      containers:
      - name: agent
        image: myregistry/agents:memory-v1.0.0
        ports:
        - containerPort: 8000

        volumeMounts:
        - name: agent-storage
          mountPath: /app/agent-state

  # Persistent volume for agent state
  volumeClaimTemplates:
  - metadata:
      name: agent-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 10Gi
```

Each replica gets its own persistent volume (memory-agent-0, memory-agent-1, etc).

---

## 3. Horizontal Pod Autoscaler (HPA)

Agents that receive variable load need autoscaling.

### CPU-based Autoscaling

```yaml
# k8s/agent-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: research-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: research-agent

  minReplicas: 3
  maxReplicas: 20

  metrics:
  # Scale based on CPU usage
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 75  # Scale up when avg CPU > 75%

  # Scale based on memory usage
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80  # Scale up when avg memory > 80%

  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Percent
        value: 100       # Double replicas on scaleup
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50        # Reduce by 50% on scaledown
        periodSeconds: 60
```

### Custom Metrics Autoscaling

For agents, you might want to scale based on custom metrics like "pending requests" or "LLM latency":

```python
# Expose custom metrics to Prometheus
from prometheus_client import Counter, Gauge, Histogram
import time

# Track pending requests
pending_requests = Gauge('agent_pending_requests', 'Requests waiting for LLM')

# Track LLM latency
llm_latency = Histogram(
    'agent_llm_latency_seconds',
    'Time waiting for LLM response',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

# Track tool calls
tool_calls = Counter(
    'agent_tool_calls_total',
    'Total tool calls made',
    ['tool_name']
)

def handle_request(request):
    pending_requests.inc()

    try:
        with llm_latency.time():
            response = call_llm(request)

        tool_calls.labels(tool_name="web_search").inc()
        return response

    finally:
        pending_requests.dec()
```

Then create HPA based on custom metric:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-custom-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: research-agent

  minReplicas: 3
  maxReplicas: 20

  metrics:
  # Scale based on pending requests (custom metric)
  - type: Pods
    pods:
      metric:
        name: agent_pending_requests
      target:
        type: AverageValue
        averageValue: "5"  # Scale up if avg pending > 5 per pod

  # Scale based on LLM latency
  - type: Pods
    pods:
      metric:
        name: agent_llm_latency_seconds
      target:
        type: AverageValue
        averageValue: "1"  # Scale up if avg latency > 1 second
```

---

## 4. Load Balancing Strategies

### Round-Robin (Default)

```
Request 1 → Agent Pod 1
Request 2 → Agent Pod 2
Request 3 → Agent Pod 3
Request 4 → Agent Pod 1  (cycles back)
```

Works fine when all agents have equal capacity. Default K8s behavior.

### Least Connections

Route to the pod with fewest active connections:

```yaml
# Requires service mesh (Istio) to implement
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: agent-lb
spec:
  host: research-agent-service
  trafficPolicy:
    loadBalancer:
      simple: LEAST_CONN  # Route to least-connected pod
```

Better for agents since long-running requests shouldn't block other clients.

### GPU-Aware Load Balancing

If agents use different GPUs, be smart about routing:

```python
# Custom load balancer logic
class GPUAwareRouter:
    def __init__(self, agents: dict[str, AgentPod]):
        """
        agents: {
            "pod-1": AgentPod(gpu="A100-0", utilization=0.3),
            "pod-2": AgentPod(gpu="A100-1", utilization=0.8),
        }
        """
        self.agents = agents

    def select_pod(self, request):
        """Route to agent with lowest GPU utilization"""
        best_pod = min(
            self.agents.items(),
            key=lambda x: x[1].gpu_utilization
        )
        return best_pod[0]

    def get_metrics(self):
        """Expose GPU metrics to Prometheus"""
        for pod_name, agent in self.agents.items():
            yield f'agent_gpu_utilization_percent{{pod="{pod_name}"}} {agent.gpu_utilization * 100}'
```

### Session Affinity (Sticky Sessions)

For agents with memory/context, keep user sessions on same pod:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: stateful-agent-service
spec:
  selector:
    app: stateful-agent

  sessionAffinity: ClientIP  # Route all requests from same client to same pod
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 86400  # 24 hours - keep session alive

  ports:
  - port: 80
    targetPort: 8000
```

Client IP is used as session key, ensuring all requests from a client go to the same pod.

---

## 5. Cost Optimization

### Using Spot Instances

For non-critical agent workloads, use cheaper spot instances:

```yaml
# k8s/spot-agent-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: research-agent-spot
spec:
  replicas: 5  # More replicas since they're cheap but unreliable

  template:
    spec:
      # Tolerate spot instance interruptions
      tolerations:
      - key: cloud.google.com/gke-preemptible
        operator: Equal
        value: "true"
        effect: NoExecute

      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            nodeAffinityTerm:
              matchExpressions:
              - key: cloud.google.com/gke-preemptible
                operator: In
                values: ["true"]  # Prefer spot instances

      containers:
      - name: agent
        image: myregistry/agents:v1.2.3
```

**Cost calculation:**
- Regular instance: $0.50/hour × 24 = $12/day
- Spot instance: $0.10/hour × 24 = $2.40/day (20% of cost)
- Trade-off: Can be interrupted anytime, so need graceful shutdown

### Right-Sizing GPU Instances

```yaml
# Pod requesting 1 GPU
spec:
  containers:
  - name: agent
    resources:
      requests:
        nvidia.com/gpu: "1"
      limits:
        nvidia.com/gpu: "1"
```

**GPU options by workload:**

| Workload | GPU Type | Cost | When to Use |
|----------|----------|------|------------|
| **LLM inference** | A100 80GB | High ($2/hr) | Large models (70B+), high throughput |
| **LLM inference** | L40S | Medium ($0.70/hr) | Medium models (13B-70B), good balance |
| **LLM inference** | T4 | Low ($0.35/hr) | Small models (7B), batch inference |
| **Embedding** | T4 | Low ($0.35/hr) | Fast enough, low cost |
| **Tool execution** | None | None | Pure CPU agents without GPU |

Right-size based on model:

```python
# Auto-select GPU based on model
GPU_REQUIREMENTS = {
    "llama-7b": "T4",
    "llama-13b": "L40S",
    "llama-70b": "A100",
    "embedding-small": "CPU",
}

def get_pod_spec(agent_version: str, model: str):
    gpu_type = GPU_REQUIREMENTS[model]

    if gpu_type == "CPU":
        return {"cpu": "4", "memory": "8Gi"}  # No GPU

    return {
        "gpu": f"nvidia.com/{gpu_type}": 1,
        "cpu": "8",
        "memory": "32Gi"
    }
```

### Model Routing for Cost Optimization

Route requests to cheapest model that works:

```python
class CostAwareRouter:
    """Route to appropriate model based on request complexity"""

    def route_request(self, request: str) -> dict:
        complexity = self.estimate_complexity(request)

        if complexity < 0.3:  # Simple request
            model = "llama-7b"  # Cheapest
            endpoint = "http://llama7b-service:8000"
        elif complexity < 0.7:  # Medium
            model = "llama-13b"
            endpoint = "http://llama13b-service:8000"
        else:  # Complex
            model = "llama-70b"  # Most expensive but best quality
            endpoint = "http://llama70b-service:8000"

        response = call_model(request, endpoint)

        # Log which model was used
        log_metric("model_used", model)
        log_metric("cost", MODEL_COSTS[model])

        return response

    def estimate_complexity(self, request: str) -> float:
        """Complexity from 0 (simple) to 1 (complex)"""
        # Check for indicators of complexity
        indicators = {
            "requires reasoning": 0.4,
            "multiple steps": 0.3,
            "domain specific": 0.2,
            "math intensive": 0.3,
        }
        return sum(indicators.values() if indicator in request.lower()
                   for indicator in indicators.keys()) / len(indicators)
```

Cost savings:

```
100 requests/day:
- All routed to Llama 70B: 100 × $2/hr ÷ 24 ÷ 10 = $0.83/day
- Routed smartly:
  - 30% to Llama 7B: 30 × $0.35/hr ÷ 24 ÷ 10 = $0.04/day
  - 50% to Llama 13B: 50 × $0.70/hr ÷ 24 ÷ 10 = $0.15/day
  - 20% to Llama 70B: 20 × $2/hr ÷ 24 ÷ 10 = $0.17/day
- Total: $0.36/day (57% savings)
```

---

## 6. High Availability Patterns

### Health Checks

Agents must be monitored at multiple levels:

```python
# agent_service.py - Implement /health and /ready endpoints

from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

class HealthStatus:
    def __init__(self):
        self.start_time = datetime.now()
        self.nim_accessible = True
        self.last_request_time = None

health = HealthStatus()

@app.get("/health")
async def health_check():
    """Liveness probe - is agent alive?"""
    uptime = (datetime.now() - health.start_time).total_seconds()

    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/ready")
async def readiness_check():
    """Readiness probe - can agent handle requests?"""

    checks = {
        "nim_accessible": health.nim_accessible,
        "memory_available": psutil.virtual_memory().percent < 90,
        "requests_pending": len(pending_requests) < 100
    }

    all_ready = all(checks.values())

    return {
        "ready": all_ready,
        "checks": checks
    }

@app.post("/query")
async def query(request: QueryRequest):
    """Process agent request"""
    health.last_request_time = datetime.now()

    try:
        response = await agent.process(request)
        return response
    except Exception as e:
        health.nim_accessible = False
        raise
```

### Graceful Shutdown

Agents need time to finish processing requests before terminating:

```python
# Handle Kubernetes termination signal
import signal
import asyncio

class AgentServer:
    def __init__(self):
        self.is_shutting_down = False
        self.active_requests = set()

    def handle_shutdown(self, signum, frame):
        """Called when K8s sends SIGTERM"""
        print("Shutdown signal received, gracefully terminating...")
        self.is_shutting_down = True

        # Wait for active requests to complete (max 30s)
        start = time.time()
        while self.active_requests and time.time() - start < 30:
            print(f"Waiting for {len(self.active_requests)} requests to finish...")
            time.sleep(1)

        if self.active_requests:
            print("Timeout reached, forcing shutdown")

        exit(0)

    async def process_request(self, request):
        if self.is_shutting_down:
            return {"error": "Server shutting down"}

        request_id = uuid4()
        self.active_requests.add(request_id)

        try:
            return await self.agent.query(request)
        finally:
            self.active_requests.discard(request_id)

# Register signal handler
server = AgentServer()
signal.signal(signal.SIGTERM, server.handle_shutdown)
```

Kubernetes respects `terminationGracePeriodSeconds` to allow graceful shutdown:

```yaml
spec:
  terminationGracePeriodSeconds: 30  # Wait up to 30s for graceful shutdown
  containers:
  - name: agent
    lifecycle:
      preStop:
        exec:
          command: ["sh", "-c", "sleep 5"]  # Give load balancer time to stop routing
```

### Pod Disruption Budgets

Ensure availability during cluster maintenance:

```yaml
# k8s/agent-pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: research-agent-pdb
spec:
  minAvailable: 2  # Always keep at least 2 pods running
  selector:
    matchLabels:
      app: research-agent
```

Without PDB, K8s could drain a node and take down all 3 replicas. With PDB, at least 2 stay running.

### Multi-Region Failover

For critical agents, replicate across regions:

```yaml
# us-east (Primary)
agents-us-east-1:
  replicas: 5
  region: us-east-1

agents-us-west:
  replicas: 3
  region: us-west-2

# Route to primary, failover to secondary
apiVersion: v1
kind: Service
metadata:
  name: research-agent-global
spec:
  externalTrafficPolicy: Cluster

  # DNS round-robin across regions
  externalDNS:
    - agents-us-east-1.internal
    - agents-us-west-2.internal
```

---

## 7. Monitoring & Observability

### Key Metrics for Agent Containers

```python
from prometheus_client import Counter, Gauge, Histogram

# Request metrics
request_count = Counter(
    'agent_requests_total',
    'Total requests processed',
    ['agent', 'status']
)

request_latency = Histogram(
    'agent_request_duration_seconds',
    'Request latency',
    ['agent'],
    buckets=[0.1, 0.5, 1, 2, 5, 10]
)

# Resource metrics
container_memory = Gauge(
    'agent_memory_usage_bytes',
    'Memory used by agent container'
)

container_cpu = Gauge(
    'agent_cpu_usage_percent',
    'CPU used by agent container'
)

# LLM metrics
llm_calls = Counter(
    'agent_llm_calls_total',
    'Total LLM API calls',
    ['model', 'status']
)

llm_tokens = Counter(
    'agent_llm_tokens_total',
    'Total tokens sent to/from LLM',
    ['direction']  # 'input' or 'output'
)
```

### Observability Stack

```yaml
# k8s/monitoring-stack.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: monitoring

---
# Prometheus - collect metrics
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
spec:
  template:
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        args:
          - "--config.file=/etc/prometheus/prometheus.yml"
          - "--storage.tsdb.path=/prometheus"

---
# Grafana - visualize metrics
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  template:
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        ports:
        - containerPort: 3000

---
# Loki - collect logs
apiVersion: v1
kind: ConfigMap
metadata:
  name: loki-config
  namespace: monitoring
data:
  loki-config.yaml: |
    auth_enabled: false
    ingester:
      chunk_idle_period: 3m
    limits_config:
      enforce_metric_name: false
```

---

## Exam-Focused Summary

**Containerization for agents:**
- Docker multi-stage builds save layer size
- ENV variables let you configure NIM/LLM endpoints at runtime
- Health checks are critical (agents hang on external service failures)

**Kubernetes deployments:**
- Deployment for stateless agents (default)
- StatefulSet for agents with persistent memory
- Init containers wait for dependencies (NIM, databases)

**Scaling and load balancing:**
- HPA scales on CPU, memory, or custom metrics
- Least-connections better than round-robin for long requests
- Spot instances 80% cheaper, but need graceful shutdown

**Cost optimization:**
- Right-size GPUs by model requirements
- Model routing (small → cheap, complex → expensive)
- Spot instances + Pod Disruption Budgets = reliable + cheap

**High availability:**
- Always implement /health and /ready endpoints
- Graceful shutdown waits for pending requests
- Pod Disruption Budgets protect against cluster maintenance

---

## Exam Practice Questions

**Question 1:** You have a research agent that calls Llama 70B via NIM. The agent hangs occasionally when NIM is unavailable. What should you add to the Deployment?

A) Higher CPU limits
B) Init container checking NIM connectivity, /ready endpoint checking NIM health
C) More replicas
D) Increase terminationGracePeriodSeconds

**Correct:** B - You need both a dependency check before pod starts (init container) and a continuous health check (/ready) that stops sending traffic to unhealthy pods.

---

**Question 2:** Your agent cluster costs $5000/month. CEO asks for 50% cost reduction. What's the best approach?

A) Upgrade to faster GPUs to process more requests
B) Right-size models (simple queries → small model) + use spot instances for non-critical agents
C) Reduce replicas from 5 to 3
D) Use CPU-only instances instead of GPUs

**Correct:** B - You need both intelligent routing (model selection) AND cheaper infrastructure (spot instances). Right model for request type = fewer resources needed.

---

**Question 3:** Agent Pod A and Pod B both handle requests. Pod A has 20 active connections, Pod B has 2. What load balancing should you use?

A) Round-robin (default)
B) Least-connections
C) Random
D) Session affinity

**Correct:** B - Least-connections routes new requests to Pod B, balancing the load. Round-robin would send to Pod A next, making imbalance worse.

---

## Key Takeaways

1. **Multi-stage Docker builds** minimize agent container size while maintaining all dependencies
2. **Init containers** handle dependency ordering (wait for NIM before starting agent)
3. **Health checks are mandatory** - agents hang when external services fail
4. **HPA scales agents like any service**, but custom metrics (pending requests, LLM latency) work better than CPU/memory
5. **Model routing cuts costs** - route simple queries to small models, complex to large models
6. **Spot instances save 80%** - fine for non-critical agents with proper shutdown handling
7. **Stateless by default** - unless agent must remember context, use Deployment not StatefulSet
8. **Graceful shutdown protects quality** - let in-flight requests complete before terminating

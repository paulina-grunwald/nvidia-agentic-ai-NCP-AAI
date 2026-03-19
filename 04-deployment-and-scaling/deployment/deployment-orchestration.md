# Deployment Orchestration: Kubernetes vs Slurm

## Quick Comparison

| System | Full Name | Primary Purpose | Workload Type | NVIDIA Use |
|--------|-----------|---|---|---|
| **Kubernetes (K8s)** | Container orchestration platform | Microservices, web apps, APIs | Long-running services | NIM, Triton serving, APIs |
| **Slurm** | Simple Linux Utility for Resource Management | HPC batch job scheduling | Batch/scheduled jobs | DGX clusters, large-scale training |

---

## Key Differences

| Aspect | Kubernetes | Slurm |
|--------|-----------|-------|
| **Designed for** | Microservices, web apps, APIs | HPC, batch jobs, training |
| **Workload model** | Long-running services (always on) | Batch jobs (run then exit) |
| **Scaling** | Auto-scale containers based on load | Schedule jobs on GPU nodes by queue |
| **State management** | Keeps services running indefinitely | Runs job, exits when complete |
| **Deployment** | Containerized (Docker/OCI) | Process-based (direct GPU access) |
| **Use case fit** | Production inference APIs | High-performance computing |

---

## When to Use Each

### Use Kubernetes When:

- **Deploying inference APIs** - NIM services, Triton endpoints, LLM APIs
- **Building microservices** - Need service discovery, load balancing, auto-healing
- **Serving models in production** - Long-running services with multiple replicas
- **Rolling updates required** - Update models without downtime
- **Multi-model serving** - Different models on different container instances

**Example:**
```bash
Deploy 3 replicas of Llama 70B via NIM
Auto-scale to 5 replicas when latency > 500ms
Update models without taking service offline
```

### Use Slurm When:

- **Training large models** - Distributed training across GPU nodes
- **Batch processing** - Process datasets, run evals, generate predictions
- **High-performance computing** - Need direct GPU access, minimal overhead
- **Research clusters** - Academic/research setups with shared resources
- **DGX clusters** - NVIDIA DGX SuperPODs and similar HPC systems

**Example:**
```bash
sbatch train_llama.sh  # Submit job to queue
#SBATCH --gpus-per-node=8
#SBATCH --nodes=4
# Job runs on 32 H100 GPUs, exits when complete
```

---

## NVIDIA Platform Context

### What Uses Kubernetes

| Component | Purpose |
|-----------|---------|
| **NVIDIA NIM (Inference Microservices)** | Deployed as K8s containers for serving |
| **Triton Inference Server** | K8s deployment for production inference |
| **Multi-model serving** | Scale different models independently |
| **Cloud deployment** | GCP, AWS, Azure - all use K8s |
| **Enterprise deployment** | Microservices architecture |

### What Uses Slurm

| Component | Purpose |
|-----------|---------|
| **NVIDIA DGX clusters** | Native Slurm support for distributed training |
| **Large-scale training** | Model training across multiple nodes |
| **NVIDIA Base Command Platform** | Slurm-based cluster management |
| **Research deployments** | Academic and research institutions |

---

## Architecture Comparison

### Kubernetes Architecture

```
┌─────────────────────────────────────────┐
│        Kubernetes Cluster               │
├─────────────────────────────────────────┤
│  Master (Control Plane)                 │
│  - API Server                           │
│  - Scheduler                            │
│  - Controller Manager                   │
│                                         │
│  Worker Nodes                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ Pod     │  │ Pod     │  │ Pod     │ │
│  │ (NIM)   │  │ (NIM)   │  │ (Triton)│ │
│  │ GPU 0-3 │  │ GPU 0-3 │  │ GPU 0-7 │ │
│  └─────────┘  └─────────┘  └─────────┘ │
│                                         │
│  Service (Load Balancer)                │
│  → Routes requests to Pods              │
└─────────────────────────────────────────┘
```

**Key features:**
- Pods auto-restart if they fail
- Services load-balance across replicas
- Horizontal Pod Autoscaler adjusts replicas based on metrics
- Rolling updates for zero-downtime deployment

### Slurm Architecture

```
┌─────────────────────────────────────────┐
│     Slurm Cluster                       │
├─────────────────────────────────────────┤
│  Slurm Controller (Job Scheduler)       │
│  - slurmctld (controller daemon)        │
│  - Job queue management                 │
│                                         │
│  Compute Nodes                          │
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │ slurmd   │  │ slurmd   │  │ slurmd ││
│  │ GPU 0-7  │  │ GPU 0-7  │  │GPU 0-7 ││
│  │ Process  │  │ Process  │  │Process ││
│  │ (job 42) │  │ (job 43) │  │(job 44)││
│  └──────────┘  └──────────┘  └────────┘│
│                                         │
│  Job Queue (FIFO or Priority)           │
│  - Job 45 (waiting...)                  │
│  - Job 46 (waiting...)                  │
└─────────────────────────────────────────┘
```

**Key features:**
- Jobs run to completion then exit
- FIFO or priority-based job queue
- Direct GPU access per job (no containerization overhead)
- Resource reservation prevents job conflicts

---

## Operational Workflow

### Kubernetes Deployment

```yaml
# Define a Kubernetes deployment for NIM
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-70b-nim
spec:
  replicas: 3  # Start with 3 replicas
  selector:
    matchLabels:
      app: llama
  template:
    metadata:
      labels:
        app: llama
    spec:
      containers:
      - name: nim
        image: nvcr.io/nim/meta/llama3-70b-instruct:latest
        resources:
          limits:
            nvidia.com/gpu: "1"  # 1 GPU per replica
        ports:
        - containerPort: 8000

---
# Auto-scale based on latency
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llama-scaler
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama-70b-nim
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 80
```

**Workflow:**
1. Submit deployment → K8s scheduler places Pods on nodes
2. Service exposes Pods via load balancer
3. Traffic increases → HPA creates more replicas
4. Model update → Rolling update replaces Pods gradually
5. Pod crashes → K8s automatically restarts it

### Slurm Job Submission

```bash
#!/bin/bash
#SBATCH --job-name=train_llama_70b
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --output=training_%j.log

# This script runs on compute nodes
module load pytorch
python /path/to/train.py \
  --model llama-70b \
  --batch-size 64 \
  --epochs 3 \
  --save-dir ./checkpoints
```

**Workflow:**
1. Submit job → Job enters queue
2. Scheduler waits for 4×8=32 GPUs to become available
3. Job runs on reserved nodes
4. Upon completion, resources freed for next job
5. Logs saved, job exits

---

## Performance Characteristics

### Kubernetes

| Aspect | Characteristic |
|--------|---|
| **Startup time** | 10-30 seconds (container init) |
| **GPU allocation** | Shared pool (multiplexing possible) |
| **Network** | Container networking (overlay networks) |
| **Memory overhead** | ~500MB-1GB per container |
| **Latency** | ~50-100ms API calls (through service) |

**Best for:** Production APIs where startup time doesn't matter, serving is continuous

### Slurm

| Aspect | Characteristic |
|--------|---|
| **Startup time** | <1 second (direct process) |
| **GPU allocation** | Exclusive per job (no sharing) |
| **Network** | Host network (direct access) |
| **Memory overhead** | Minimal (direct process) |
| **Throughput** | Optimized for batch processing |

**Best for:** Batch workloads where every second of training matters, maximize compute efficiency

---

## NVIDIA-Specific Considerations

### DGX SuperPOD (Uses Slurm)

NVIDIA's flagship HPC system:
- 140 H100 GPUs across 20 DGX H100 nodes
- Managed by Slurm for job scheduling
- Used for large-scale model training
- Direct GPU access for maximum performance

**Typical workload:**
```bash
# Train 1T token model across 140 GPUs
sbatch --nodes=20 --gpus-per-node=8 train_massive.sh
# Job runs for days, then exits
```

### NIM in Production (Uses Kubernetes)

NVIDIA's inference microservices:
- Containerized models (Llama, Nemotron, etc.)
- Deploy on K8s for elasticity
- Auto-scale based on traffic
- Update models without stopping service

**Typical workload:**
```bash
# Deploy Llama 70B inference API
kubectl apply -f llama-nim-deployment.yaml
# Service immediately available, auto-scales with load
```

---

## Hybrid Approaches

Some organizations use both:

| Phase | Orchestrator | Reason |
|-------|---|---|
| **Development** | Slurm | Test training on shared DGX resources |
| **Production Training** | Slurm | Run scheduled training jobs on DGX |
| **Model Serving** | Kubernetes | Deploy trained models as APIs |
| **Inference Serving** | Kubernetes | Auto-scale serving based on traffic |

**Example pipeline:**
```
DGX Cluster (Slurm)          K8s Cluster (Production)
├─ Train model               ├─ Deploy NIM
│  └─ 48 hours              │  └─ Serve predictions
│                            │
├─ Evaluate                  ├─ A/B test
│  └─ 2 hours               │  └─ Compare accuracy
│                            │
└─ Export model              └─ Scale based on demand
   └─ Push to registry          └─ Handle traffic spikes
```

---

## Exam-Focused Summary

**Common exam patterns:**

| Scenario | Answer | Why |
|----------|--------|-----|
| "Deploy NIM for inference API" | Kubernetes | K8s is standard for serving |
| "Train model on DGX cluster" | Slurm | DGX uses Slurm natively |
| "Need auto-scaling for traffic spikes" | Kubernetes | HPA built-in |
| "Minimize GPU initialization overhead" | Slurm | Direct process, no container init |
| "Multi-model serving in production" | Kubernetes | Service mesh, load balancing |
| "Batch training across 20 GPUs" | Slurm | Optimized for batch jobs |

**Key distinction for the exam:**
- **Kubernetes = Long-running services** (inference, APIs, always-on)
- **Slurm = Batch jobs** (training, batch processing, scheduled work)


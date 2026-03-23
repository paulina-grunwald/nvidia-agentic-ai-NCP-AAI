# GPU Hardware & MIG Study Notes

---

## A100 vs H100: The Hardware Tiers

When you see a question about GPU selection, here's the core distinction:

**A100: The Workhorse**

- 40GB or 80GB memory variants
- Solid LLM performance (your baseline)
- No Transformer Engine (FP8 optimization)
- Cost-effective at ~$10K-15K per unit
- Full MIG support for partitioning

**H100: The High-Performance Tier**

- 80GB standard (no 40GB option)
- About 3x faster for LLM workloads compared to A100
- Transformer Engine with FP8 support (significant efficiency gain)
- Higher cost at ~$25K-40K per unit
- Also supports MIG, but typically used for demanding single-model workloads

### When to Pick Which?

The exam loves this distinction: don't automatically reach for H100. If your workload runs fine on A100, you save costs. The H100 shines when you're bottlenecked by compute speed or need the FP8 Transformer Engine for inference optimization.

---

## MIG (Multi-Instance GPU): The Partitioning Game

MIG is one of the most frequently tested topics. Here's what you must know cold:

### MIG Availability

Only A100 and H100 support MIG. Full stop. If you see an older GPU (V100, RTX) in a question, MIG is not an option.

### MIG Partitioning Rules

A single A100 or H100 can be split into up to 7 independent instances. Each instance gets:

- Isolated GPU memory (no sharing between instances)
- Dedicated compute cores
- Dedicated cache lines
- Independent failure domain

This complete isolation is crucial for multi-tenant scenarios.

### When MIG Solves Your Problem

You're running multiple small models simultaneously on an expensive GPU and want to prevent "noisy neighbor" effects. MIG ensures one model's spike doesn't starve another. It's also your answer when you have mixed workloads that don't justify separate physical GPUs.

### When MIG is NOT the Answer

Your application needs maximum raw performance for a single large model. If a 70B parameter model needs the full GPU's compute power, splitting the GPU actually hurts you. MIG adds minor overhead, and you lose pooled resources.

---

## GPU Decision Keywords (Exam Triggers)

When reading exam questions, watch for these keywords:

**"GPU utilization inefficiency"**
Always leads to: Profile first (with profiling tools), then consider MIG partitioning or GPU reallocation.

**"Mixed A100/H100 cluster"**
You'll need to profile each model independently to understand whether it needs H100's 3x performance boost or if A100 suffices.

**"Multiple small agents on a single GPU"**
Classic MIG scenario. Different agents can run in isolated instances.

**"Noisy neighbor" or "resource contention"**
MIG isolation is your direct answer.

**"Production GPU monitoring and alerting"**
Triggers the need for DCGM and monitoring infrastructure.

---

## Profiling Tools: The Holy Trinity

The exam distinguishes between three different profiling/monitoring tools with specific use cases:

### Nsight Systems

- When question mentions: "kernel profiling," "timeline analysis," "debug GPU performance," "understand execution flow"
- Purpose: Detailed kernel-level profiling with timeline visualization
- Use case: Debugging performance bottlenecks and understanding GPU kernel execution order
- Output: Detailed timeline showing CPU/GPU activity, kernel calls, memory transfers

### DCGM (Data Center GPU Manager)

- When question mentions: "monitor production GPUs," "utilization metrics," "alerts," "continuous monitoring," "health checks"
- Purpose: Real-time GPU monitoring and telemetry collection in production environments
- Use case: Alerting on overutilization, temperature anomalies, power consumption
- Integration: Works with Grafana for dashboards

### Triton Model Analyzer

- When question mentions: "optimize batch size," "find best concurrency," "serving configuration," "inference performance tuning"
- Purpose: Benchmarking inference configurations under different batch sizes and concurrency levels
- Use case: Determining optimal settings for production model serving
- Output: Recommendations for batch size, concurrency, latency/throughput tradeoffs

The key distinction: Nsight Systems = development-time debugging, DCGM = production monitoring, Triton Model Analyzer = inference optimization.

---

## GPU Allocation Strategy

The exam often tests whether you avoid these common traps:

**The Overprovisioning Trap**
Never default to "deploy everything on H100s because they're fastest." This wastes money. Instead, profile your models and match them to appropriate tiers. An 8B chatbot on H100 is expensive overkill.

**The Random Distribution Trap**
Don't randomly scatter workloads across available GPUs. Profile first, then assign based on computational needs.

**The CPU-First Trap**
Don't suggest CPU deployment as a cost-saving measure without understanding that GPU deployment might actually be cheaper when you factor in the hardware you'd need to buy for CPU scaling.

**Smart Strategy**

1. Profile each model/agent to understand its compute and memory needs
2. Categorize workloads by performance sensitivity
3. Assign A100 to standard workloads, H100 to performance-critical ones
4. Use MIG to pack multiple non-competing workloads onto expensive GPUs

---

## Nemotron Model Variants

NVIDIA's Nemotron model family comes in different sizes, each with distinct deployment profiles:

| Variant       | Parameter Count | Best For                                       | GPU Tier               |
| ------------- | --------------- | ---------------------------------------------- | ---------------------- |
| Nemotron-8B   | 8 billion       | Simple chatbots, edge devices, quick inference | A100 (40GB) or smaller |
| Nemotron-22B  | 22 billion      | Balanced tasks with moderate reasoning         | A100 (40GB) or H100    |
| Nemotron-70B  | 70 billion      | Complex reasoning, multi-step agents           | H100 or multiple A100s |
| Nemotron-340B | 340 billion     | Enterprise reasoning, highest quality outputs  | Multiple H100s         |

All Nemotron variants are deployed as NIM containers (NVIDIA Inference Microservices). They expose a standardized API endpoint, which means you can swap model sizes without rewriting your application code.

The size matters because it directly impacts GPU memory requirements and computational load. An 8B model and a 340B model on the same H100 won't both run at full efficiency due to memory constraints.

---

## Confidential Computing

A newer topic gaining prominence in the exam: NVIDIA's confidential computing support on Hopper and Blackwell architectures.

**What it does:** Hardware-level encryption of GPU memory and compute. Data is encrypted at rest and in transit, visible only in plaintext to trusted software running in the Trusted Execution Environment (TEE).

**When the exam mentions it:** Look for keywords like "advanced encryption," "sensitive data processing," "regulated environments with data privacy requirements," or "hardware-backed security."

**Not to be confused with:** MIG isolation (which is resource isolation, not encryption).

---

## Exam MCQs

### Question 1

A company runs four inference services on a single expensive H100 GPU. Each service has unpredictable traffic patterns that sometimes spike. When one service gets heavy traffic, the others experience increased latency. What's the best solution?

A) Deploy each service to its own H100 GPU
B) Implement MIG to isolate each service in its own instance
C) Use spot instances to reduce cost
D) Implement dynamic batching in Triton

**Correct Answer: B**

Dynamic batching (D) improves throughput, not isolation. Spot instances (C) don't solve the noisy neighbor problem. Option A is expensive overkill. Option B directly solves resource contention through complete hardware isolation.

---

### Question 2

You're profiling a production deployment and need to understand why GPU utilization is only 45% despite high incoming request volume. Your DevOps team also wants automated alerts if utilization drops below 40%. What tools do you need?

A) Nsight Systems for monitoring, DCGM for alerts
B) DCGM for monitoring, Triton Model Analyzer for alerts
C) Triton Model Analyzer for understanding utilization patterns
D) Nsight Systems for production monitoring

**Correct Answer: A**

Nsight Systems is development-focused. Triton Model Analyzer doesn't handle production alerting. DCGM is the production monitoring tool with alert capabilities. You'd use Triton Model Analyzer in the next step to test configuration changes.

---

### Question 3

Your team has a Nemotron-340B model and a Nemotron-8B chatbot. They have completely different performance requirements. Which statement is true?

A) Both should run on H100 for consistency
B) The 340B requires multiple GPUs; the 8B can share A100 space with other models via MIG
C) Both should run on A100 since they're the same model family
D) The 8B should run on H100 for faster response times

**Correct Answer: B**

The 340B's massive parameter count requires multiple high-end GPUs. The 8B is lightweight and perfect for MIG partitioning alongside other small models on an A100. Same model family doesn't mean same hardware requirements.

---

### Question 4

You're designing a cost-optimized inference cluster for a mix of workloads: Nemotron-70B (latency-sensitive), Nemotron-22B (standard), and Nemotron-8B (batch processing). What allocation strategy makes sense?

A) Deploy all on A100 to save costs
B) Deploy all on H100 for consistent performance
C) Profile each model, deploy 70B on H100, 22B on A100, partition 8B via MIG
D) Deploy randomly across available GPUs for load balancing

**Correct Answer: C**

Profile-driven allocation matches models to appropriate hardware. The latency-sensitive 70B benefits from H100's speed. The 22B works fine on A100. The 8B can share space via MIG. Random distribution (D) is inefficient. All-A100 (A) leaves 70B underperforming. All-H100 (B) wastes money.

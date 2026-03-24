# Exam Decision Rules & Patterns
## NVIDIA NCP-AAI Certification Exam Prep

This document captures the subtle decision patterns and keyword triggers that don't fit neatly into technical topics. Think of it as your "last mile" cheat sheet for exam day.

---

## 1. Compliance Deployment: Watch the Verb

Compliance questions are heavily tested. The key is recognizing the language shift between strict and flexible requirements.

### Strict Compliance Language
Keywords: "ensure compliance," "guarantee," "strict regulations," "zero risk," "compliance-critical"

Answer pattern: On-premises deployment or strict MIG isolation
- Data never leaves your building in on-prem
- MIG isolation within your control means no resource sharing with external tenants
- Both options keep compliance data under your direct governance

### Balanced Compliance Language
Keywords: "balance compliance," "optimize costs," "practical approach," "acceptable risk level," "reasonable security"

Answer pattern: Hybrid cloud approach
- Some data can safely move to cloud (analytical, non-regulated)
- Sensitive data stays on-premises
- Cost optimization becomes a valid consideration alongside compliance

### NVIDIA AI Enterprise Trigger
When NVIDIA AI Enterprise is mentioned alongside compliance, the exam often leans toward NVIDIA-specific solutions like MIG isolation rather than generic cloud compliance. NVIDIA AI Enterprise includes specific compliance tooling.

### Core Logic
On-premises = maximum control = highest compliance. Hybrid = managed risk. Pure cloud = most risk (even if the cloud provider is secure, data travels outside your network).

---

## 2. Slurm vs Kubernetes: Different Jobs, Different Orchestrators

This distinction trips up many candidates because both run on GPU clusters.

### Kubernetes
- Purpose: Container orchestration for microservices
- Use case: NIM (inference APIs), autoscaling, rolling updates, distributed microservices
- Strength: Service discovery, load balancing, horizontal scaling of stateless containers
- API pattern: REST/gRPC services with standard Kubernetes scaling

### Slurm
- Purpose: HPC batch job scheduler
- Use case: Large-scale GPU training, batch processing, DGX SuperPOD deployments
- Strength: Fine-grained job scheduling, GPU allocation primitives, multi-job coordination
- API pattern: sbatch job submission, job arrays

### Decision Rule
"NIM deployment with autoscaling" → Kubernetes (NIM exposes HTTP APIs that Kubernetes understands).
"Large-scale LLM training on DGX" → Slurm (training is batch work, not request-response).
"Multiple independent agents serving requests" → Kubernetes (stateless service instances).
"GPU cluster running training pipelines" → Slurm (batch scheduling makes sense).

The wrong answer often mixes these: deploying a training job to Kubernetes's service autoscaler misses that training is batch work, not request response.

---

## 3. Deployment Architecture Rules

These rules apply across inference, scaling, and HA scenarios:

### Multi-Zone Deployments
"Multiple zones" or "high availability" = replicate your inference service across geographically separated zones. This protects against zone failures.

### Scaling Metrics
Always scale on GPU-specific metrics, never CPU or memory:
- GPU utilization percentage
- GPU memory used
- Queue depth of pending requests to GPU

Scaling on CPU when your bottleneck is GPU misses the actual constraint.

### Spot Instances
Spot = low cost but can be reclaimed without warning
- DON'T use when: "zero downtime required," "critical service," "guaranteed availability"
- DO use when: batch jobs, development/testing, workloads with graceful degradation

### Dynamic Batching in Triton
Triton can wait for multiple requests to arrive, then batch them into a single GPU invocation. This improves throughput without hurting latency as much as naive batching would.
- Trigger phrase: "optimize throughput," "group incoming requests"
- Not the answer for: single-client requests, ultra-low latency requirements

### Zero Downtime Model Swaps
When you need to update a model without dropping requests:
- NIM provides standardized API endpoints
- You deploy a new model container in Kubernetes
- Kubernetes routes new requests to new container, drains old requests from old container
- Client code never changes because the API is identical

### Horizontally Scalable Inference
"Scale to handle concurrent requests" typically means:
1. Deploy model on Kubernetes with NIM
2. Set up horizontal pod autoscaling based on GPU metrics
3. Add load balancer in front
4. Each pod is independent (stateless)

### Stateful Scaling: Dialogue Continuity
"Maintain dialogue continuity" or "session state across multiple requests" = distributed state problem

Solution pattern:
- Multiple inference instances need access to shared conversation state
- Use Redis or AIQ shared memory layer
- NEVER use sticky sessions (route user to same instance) because that single instance becomes a bottleneck and failure point

Sticky sessions are actually an anti-pattern in modern designs because they prevent horizontal scaling.

---

## 4. Concurrent Requests + Low Latency

This is a common "trap" question pattern:

### The Problem
You have many concurrent requests arriving simultaneously and need sub-100ms latency.

### Wrong Answer
"Use a single GPU with large batch size"
- Large batch size increases throughput but INCREASES latency (p99 latency gets worse)
- You can't serve 10 concurrent requests in one batch without delaying some of them

### Right Answer
"Multiple GPU instances with dynamic batching"
- Each GPU gets small batches (low latency)
- Across instances, you serve high concurrency
- Dynamic batching helps without the latency penalty of naive batching

Key insight: Throughput and latency are different dimensions. Batch size improves throughput at the cost of latency.

---

## 5. Framework Swap Trap

The exam frequently tempts you with framework changes:

Example: "Your LangChain agent is struggling with memory management. Should you switch to LlamaIndex or AutoGen instead?"

**Answer: No. Fix the actual problem with better tooling in your current framework.**

The exam loves this because it tests whether you understand the root cause vs symptom swapping. Switching from AutoGen to CrewAI doesn't fix memory leaks. Fixing memory management does.

Only switch frameworks if the current one fundamentally lacks required capability (not a configuration/tuning issue).

---

## 6. NVIDIA-Native Preference

This is subtle but pervasive in the exam: NVIDIA questions strongly favor the NVIDIA-native stack.

### NVIDIA-Native Stack
- Agent framework: NeMo agents
- Toolkit: AIQ Toolkit
- Inference: NIM (standardized containers)
- Model serving: Triton Inference Server
- Safety: Guardrails
- Monitoring: DCGM + Grafana

### Non-NVIDIA Options (Usually Wrong)
- LangChain
- LlamaIndex
- Redis (for agent coordination)
- Custom Python orchestration

When both options are technically viable, the exam leans toward NVIDIA solutions. This isn't a guarantee, but it's a strong pattern.

Example: "How do you manage multi-agent state?"
- Generic answer: Redis
- NVIDIA-preferred answer: AIQ Toolkit (NVIDIA-native state management)

---

## 7. Chain of Thought (CoT) Special Cases

CoT is the sequential reasoning pattern. The exam tests subtle boundary conditions:

### CoT Effectiveness Threshold
CoT only produces reliable reasoning chains on large models (~100B+ parameters). Smaller models can generate plausible-sounding but incorrect reasoning steps. Don't use CoT with 8B models for complex reasoning.

### CoT + Memory = Adaptation
"Adapt reasoning based on previous interactions" = CoT reasoning combined with persistent memory of past exchanges. This is different from basic CoT because it learns from conversation history.

### CoT is NOT ReAct
- CoT: Step-by-step reasoning, sequential thinking
- ReAct: Reason + Act + Observe loop, interleaves reasoning with action

Example: "Agent breaks problem into steps" = CoT (thinking-focused).
Example: "Agent reasons, calls tool, observes result, reasons again" = ReAct (action-focused).

### CoT CAN Use Tools
A common misconception: CoT and tool use are separate. They're not. CoT plans tool calls as reasoning steps. A reasoning chain might be: "To solve this, I need to query the database (step 1), analyze the results (step 2), then synthesize (step 3)." Each step might involve tool calls.

---

## 8. Circuit Breaker States

Circuit breaker pattern manages service failure:

**CLOSED State**: Normal operation, requests pass through
**Failure threshold exceeded** → **OPEN State**: Block all requests immediately
**Timeout expires** → **HALF-OPEN State**: Allow one test request through
**Test request succeeds** → **CLOSED State**: Resume normal operation
**Test request fails** → **OPEN State**: Stay open, wait for timeout again

### Exam Trap
Many candidates forget HALF-OPEN or try to skip straight from OPEN to CLOSED. HALF-OPEN is the recovery testing phase. Never skip it.

### Keyword Trigger
"Fail fast," "circuit breaker," "cascading failures" = circuit breaker pattern applies.

---

## 9. HITL vs NOT HITL

Human-in-the-Loop (HITL) is frequently tested with subtle definitions:

### YES, This is HITL
- Agent pauses and waits for human approval before proceeding
- Example: "Agent selects action, human reviews and approves, agent executes"
- Characteristic: Synchronous blocking on human input

### NO, This is NOT HITL
- Agent completes task and writes output
- Human reviews output asynchronously later
- Example: "Agent finishes reasoning, presents results, human reads and validates"
- Characteristic: Asynchronous review

The key difference: Does the agent's execution depend on human decision-making, or does the human just validate the output afterward?

HITL questions often frame it as: "The agent pauses while waiting for user input" vs "The agent finishes and displays the result."

---

## 10. Subproblems vs Steps

This distinction determines the algorithm:

### "Break into Subproblems"
Signals: Tree of Thought (ToT) pattern
- Exploring different solution paths through a problem space
- Evaluating which branch is most promising
- Backtracking if a branch fails
- Example: "Multiple ways to approach the problem, agent explores promising paths"

### "Break into Steps"
Signals: Chain of Thought (CoT) pattern
- Sequential reasoning
- One step after another
- No branching
- Example: "First analyze, then retrieve data, then synthesize"

### "Break into Tasks"
Signals: ReAct pattern
- Reason → Act → Observe loop
- Agent executes actions as part of thinking
- Example: "Agent reasons about what data it needs, queries database, observes results, continues reasoning"

### Keyword Differentiation
- "How does the agent THINK about this?" = CoT (reasoning chain)
- "How does the agent EXPLORE solutions?" = ToT (subproblems, tree search)
- "How does the agent ACCOMPLISH this?" = ReAct (action-focused)

---

## Exam MCQs

### Question 1
Your company stores encrypted patient medical records. Compliance regulations strictly require that data cannot leave the facility. You're building an inference service for clinical decision support using Nemotron-70B. What deployment approach satisfies compliance?

A) Deploy NIM containers on Kubernetes in a public cloud with RBAC controls
B) Deploy on-premises with full control over GPU infrastructure
C) Deploy on a hybrid cloud where sensitive data stays on-premises and metadata goes to cloud
D) Deploy on AWS with NVIDIA confidential computing enabled

**Correct Answer: B**

Strict compliance ("cannot leave facility") eliminates any cloud option, including hybrid (C). Public cloud (A) violates the requirement. Confidential computing (D) encrypts data but data still travels to cloud infrastructure. On-premises (B) guarantees data never leaves.

---

### Question 2
You're deploying a multi-agent system that requires coordinated state management across multiple inference instances. Which statement is true?

A) Use sticky sessions to route each user to the same instance
B) Use Redis for distributed state, avoid sticky sessions
C) Use NIM's internal state synchronization
D) Use AIQ Toolkit's distributed state management with multiple instances

**Correct Answer: D**

Sticky sessions (A) prevent horizontal scaling and create bottlenecks. Redis (B) is generic but not NVIDIA-native; on this exam, NVIDIA-native (AIQ Toolkit) is preferred. NIM doesn't have built-in state sync (C). AIQ Toolkit is the NVIDIA-native solution for multi-instance state coordination.

---

### Question 3
Your team is considering switching from LangChain agents to AutoGen because the agents occasionally produce verbose outputs that waste compute. Is this a good decision?

A) Yes, AutoGen has built-in output optimization
B) Yes, different frameworks solve different problems
C) No, the issue is verbosity tuning, not framework limitations
D) Yes, AutoGen is lighter weight

**Correct Answer: C**

Verbose output is a prompt/configuration issue, not a framework limitation. Both LangChain and AutoGen can be tuned for conciseness. Switching frameworks doesn't fix the root cause. Option C recognizes the actual problem.

---

### Question 4
Your inference service experiences sudden spikes in requests. Currently, you scale based on CPU utilization, but GPUs hit 99% utilization while CPUs remain at 45%. What should you change?

A) Add more CPU cores
B) Scale based on GPU utilization instead of CPU
C) Use spot instances to handle spike traffic
D) Implement batch size increases

**Correct Answer: B**

The bottleneck is clearly GPU, not CPU. Adding CPU (A) doesn't help. Spot instances (C) are for cost optimization, not performance. Batch increases (D) worsen latency. GPU-based scaling (B) addresses the actual constraint.

---

### Question 5
An agent needs to solve a complex reasoning task. It explores multiple solution approaches, evaluates which looks most promising, and backtracks when an approach fails. What pattern describes this?

A) Chain of Thought
B) Tree of Thought
C) ReAct
D) Dynamic programming

**Correct Answer: B**

The key phrase is "explores multiple approaches" and "backtracks." That's tree search, not linear reasoning. Chain of Thought (A) is sequential without branching. ReAct (C) interleaves reasoning and actions. Dynamic programming (D) is irrelevant to reasoning patterns.

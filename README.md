# NVIDIA Certified Professional: Agentic AI (NCP-AAI)

Comprehensive exam notes and study resources for the [NVIDIA Agentic AI Professional Certification](https://www.nvidia.com/en-us/learn/certification/agentic-ai-professional/).

## About This Repository

This repository consolidates knowledge and best practices for building production-grade agentic AI systems on NVIDIA's platform. It covers foundational concepts through advanced enterprise deployment patterns, with emphasis on NVIDIA - specific technologies like NIMs, NeMo, Triton, and NeMo Agents.

**Who should use this?**
- Developers and architects preparing for the NCP-AAI certification
- Engineers building agent-based AI systems
- Teams deploying agentic AI at scale on NVIDIA infrastructure
- Anyone seeking deep understanding of agent architectures and NVIDIA's agentic AI stack

**What you'll find here:**
- **Agent Architecture Foundations** - Core concepts, design patterns (reactive, deliberative, hybrid, multi-agent)
- **Reasoning & Problem-Solving** - ReAct, Chain-of-Thought, Tree of Thoughts, self-reflection techniques
- **Production Patterns** - Error handling, memory management, checkpointing, evaluation frameworks
- **NVIDIA Ecosystem Deep Dives** - NIMs, NeMo, Triton, performance optimization, enterprise deployment
- **Framework Comparisons** - LangChain/LangGraph, CrewAI, AutoGen with NVIDIA integration examples
- **Advanced Topics** - RAG, vector databases, tool integration, multi-agent coordination

The notes are structured to mirror exam domains while maintaining practical utility for implementation.

## Exam Blueprint & Domain Coverage

The NCP-AAI exam covers **10 weighted domains** across 60–70 questions (120-minute time limit):

| Domain | Weight | Study Resources |
| --- | --- | --- |
| **Agent Architecture and Design** | 15% | Agent Architecture Overview, 4-Layer Architecture, Reasoning Patterns, Stateful Orchestration, Multi-Agent Communication, Memory Management, Knowledge Graphs, Human-Agent UI Design |
| **Agent Development** | 15% | LangChain, CrewAI, AutoGen, Prompt Engineering, Custom Tools & APIs, Multimodal Models, Streaming & Feedback, Decision-Making Evaluation |
| **Evaluation and Tuning** | 13% | Evaluation Techniques, Pipelines, Benchmarking, Parameter Tuning, User Feedback, Optimization |
| **Deployment and Scaling** | 13% | Triton, Enterprise Deployment, Containerization & K8s, MLOps & CI/CD, GPU Hardware & MIG |
| **Cognition, Planning, and Memory** | 10% | Memory & Planning, Adaptive Reasoning, Reasoning Patterns, LangGraph Checkpoints |
| **Knowledge Integration and Data Handling** | 10% | RAG and Tools, Vector DB Optimization, ETL & Data Quality, Real-Time Knowledge Access |
| **NVIDIA Platform Implementation** | 7% | NIMs, NeMo, TensorRT-LLM, NeMo Guardrails, AIQ Toolkit, NeMo Curator, NeMo Aligner/RLHF, RAPIDS, Component Decision Guide, Exam Decision Rules |
| **Run, Monitor, and Maintain** | 5% | Observability & Tracing, Alerting & Maintenance, Throughput vs Latency |
| **Safety, Ethics, and Compliance** | 5% | Enterprise AI Factory, Responsible AI & Auditing, Bias Detection, Compliance |
| **Human-AI Interaction and Oversight** | 5% | HITL Workflows, Escalation Protocols, Transparency, Human-Agent UI Design |

## Quick Start

**For exam preparation:**
1. Master **Agent Architecture and Design** (15%) - foundation for all other domains
2. Study **Agent Development** (15%) using the framework guides
3. Deep-dive into **Cognition, Planning, Memory** (10%) and **Knowledge Integration** (10%)
4. Review **NVIDIA Platform Implementation** (7%) and **Deployment** (13%)
5. Cover remaining domains: Evaluation, Monitoring, Safety, Human-AI Interaction

**For implementation:** Use the Framework guides as starting points, reference NVIDIA technologies for infrastructure decisions, and apply patterns from Error Handling and enterprise deployment sections.

## Study Materials by Exam Domain

### 1. Agent Architecture and Design (15%)

- [Agent Architecture Overview](01-agent-architecture-and-design/agent-architecture-overview.md) - reactive, deliberative, hybrid, learning, multi-agent, tool-augmented architectures
- [4 Layer Agent Architecture](01-agent-architecture-and-design/4-layer-agent-architecture.md) - perception, cognition, planning, execution
- [Reasoning Patterns](01-agent-architecture-and-design/reasoning-patterns.md) - ReAct, Chain-of-Thought, Tree of Thoughts, Self-Reflection
- [Error Handling](01-agent-architecture-and-design/error-handling.md) - resilience patterns and fault tolerance
- [Stateful Orchestration](01-agent-architecture-and-design/stateful-orchestration.md) - logic trees, prompt chains, multi-step reasoning
- [Multi-Agent Communication](01-agent-architecture-and-design/multi-agent-communication.md) - multi-agent communication and orchestration
- [Memory Management](01-agent-architecture-and-design/memory-management.md) - short-term and long-term memory for context retention
- [Knowledge Graphs](01-agent-architecture-and-design/knowledge-graphs.md) - knowledge graphs for relational reasoning
- [Human-Agent UI Design](01-agent-architecture-and-design/human-agent-ui-design.md) - UI patterns for human-agent interaction

### 2. Agent Development (15%)

- [Framework Comparison](02-agent-development/agent-orchiestration-frameworks.md) - AutoGen, LangGraph, CrewAI, LangChain overview
- [LangChain/LangGraph](02-agent-development/langchain.md) - stateful graphs, memory, checkpoints, NVIDIA integration
- [CrewAI](02-agent-development/crewai.md) - role-based agent teams
- [AutoGen](02-agent-development/autogen.md) - multi-agent conversations
- [Prompt Engineering](02-agent-development/prompt-engineering.md) - zero-shot, few-shot, CoT, ReAct, tool use, prompt tuning, NIM templates
- [Custom Tools and APIs](02-agent-development/custom-tools-and-apis.md) - building, connecting, and managing external tools for agents
- [Multimodal Models](02-agent-development/multimodal-models.md) - integrating text, vision, and audio models into agent systems
- [Streaming and Feedback](02-agent-development/streaming-and-feedback.md) - dynamic conversation flows, real-time streaming, feedback mechanisms
- [Decision-Making Evaluation](02-agent-development/decision-making-evaluation.md) - assessing and refining agent decision-making

### 3. Evaluation and Tuning (13%)

- [Evaluation and Tuning Techniques](03-evaluation-and-tuning/evaluation-and-tuning-techniques.md) - RAGAS, LLM-as-Judge, trajectory evaluation, faithfulness, relevance
- [Evaluation Pipelines](03-evaluation-and-tuning/evaluation-pipelines.md) - pipeline anatomy, scoring methods, NVIDIA eval tools (AIQ, NeMo Evaluator, Triton perf_analyzer), CI/CD integration
- [Benchmarking and Comparison](03-evaluation-and-tuning/benchmarking-comparison.md) - multi-dimension comparison, statistical significance, cross-task/cross-dataset testing, model routing
- [Parameter Tuning](03-evaluation-and-tuning/parameter-tuning.md) - temperature/top-p/top-k, quantization trade-offs, RAG parameters, systematic tuning methodology
- [User Feedback Integration](03-evaluation-and-tuning/user-feedback-integration.md) - structured feedback schemas, correction feedback, implicit signals, RLHF/DPO
- [Evaluation-Driven Optimization](03-evaluation-and-tuning/evaluation-driven-optimization.md) - root cause analysis, 80/20 rule, impact vs effort matrix, optimization tracking

### 4. Deployment and Scaling (13%)

**Inference & Optimization:**
- [Triton Inference Server](04-deployment-and-scaling/inference/triton-inference-server.md) - high-performance inference
- [Triton Performance Analyzer](04-deployment-and-scaling/inference/triton-performance-analyzer.md) - profiling and optimization

**Enterprise Deployment:**
- [Enterprise Deployment Pattern](04-deployment-and-scaling/deployment/enterprise-deployment-pattern.md)
- [Deployment Orchestration](04-deployment-and-scaling/deployment/deployment-orchestration.md) - Kubernetes vs Slurm, NVIDIA context

**Containerization & Scaling:**
- [Containerization and Scaling](04-deployment-and-scaling/containerization-scaling.md) - Docker, Kubernetes patterns, HPA, load balancing, GPU scheduling, cost optimization
- [MLOps and CI/CD](04-deployment-and-scaling/mlops-cicd.md) - CI/CD pipelines, artifact versioning, A/B and canary deployments, rollback strategies

**GPU Hardware & Optimization:**
- [GPU Hardware and MIG](04-deployment-and-scaling/gpu-hardware-mig.md) - A100 vs H100, MIG partitioning, Nemotron variants, profiling tools (Nsight/DCGM/Model Analyzer), confidential computing

### 5. Cognition, Planning, and Memory (10%)

- [Memory and Planning](05-cognition-planning-memory/memory-and-planning.md) - working memory, episodic memory, procedural memory, planning architectures
- [Planning and Adaptive Reasoning](05-cognition-planning-memory/planning-and-adaptive-reasoning.md) - hierarchical planning, adaptive replanning, meta-reasoning, plan verification
- See also: [Reasoning Patterns](01-agent-architecture-and-design/reasoning-patterns.md), [Memory Management](01-agent-architecture-and-design/memory-management.md), [LangChain/LangGraph](02-agent-development/langchain.md) checkpointing

### 6. Knowledge Integration and Data Handling (10%)

- [RAG and Tools](06-knowledge-integration-data/rag-and-tools.md) - embeddings, routers, chunking, retrieval methods, vector DBs, evaluation, advanced RAG patterns
- [Vector DB Optimization](06-knowledge-integration-data/vector-db-optimization.md) - HNSW, IVF indexing, GPU-accelerated search, Milvus scaling strategies
- [ETL and Data Quality](06-knowledge-integration-data/etl-and-data-quality.md) - ETL pipelines, NeMo Curator, data validation, deduplication, quality metrics
- [Real-Time Knowledge Access](06-knowledge-integration-data/real-time-knowledge-access.md) - SQL agents, API agents, streaming data integration, live knowledge retrieval

### 7. NVIDIA Platform Implementation (7%)

**Core Components:**
- [NIMs](07-nvidia-platform-implementation/nims.md) - NVIDIA Inference Microservices
- [NeMo](07-nvidia-platform-implementation/nemo.md) - large language models, fine-tuning
- [TensorRT-LLM](07-nvidia-platform-implementation/tensorrt-llm.md) - inference optimization, mixed precision, kernel fusion, in-flight batching, latency vs throughput
- [NeMo Guardrails](07-nvidia-platform-implementation/nemo-guardrails.md) - Colang 2.0, rail types (input/output/topical/retrieval), NIM integration, programmable safety
- [NeMo Agent Toolkit](07-nvidia-platform-implementation/nemo-agent-toolkit.md)
- [AIQ Toolkit](07-nvidia-platform-implementation/aiq-toolkit.md)
- [NeMo Curator](07-nvidia-platform-implementation/nemo-curator.md) - GPU-accelerated data prep, dedup, PII redaction, quality filtering
- [NeMo Aligner and RLHF](07-nvidia-platform-implementation/nemo-aligner-rlhf.md) - RLHF pipeline, reward models, PPO, DPO alignment
- [RAPIDS and cuGraph](07-nvidia-platform-implementation/rapids-cugraph.md) - GPU-accelerated data science, knowledge graph analytics

**Architecture & Decision Guides:**
- [NVIDIA Component Decision Guide](07-nvidia-platform-implementation/nvidia-component-decision-guide.md) - which tool for which job, keyword triggers, common exam traps
- [Exam Decision Rules](07-nvidia-platform-implementation/exam-decision-rules.md) - compliance patterns, Slurm vs K8s, framework traps, HITL rules, CoT special cases
- [NIMs and NeMo Integration](07-nvidia-platform-implementation/nims-and-nemo.md) - combined NIM + NeMo workflows
- [Platform Components](07-nvidia-platform-implementation/platform-components.md)
- [Architecture Patterns](07-nvidia-platform-implementation/architecture-patterns.md)
- [NVIDIA Role in Ecosystem](07-nvidia-platform-implementation/nvidia-role-in-ecosystem.md)

### 8. Run, Monitor, and Maintain (5%)

- [Domain Overview](08-run-monitor-maintain/README.md) - monitoring, observability, and maintenance roadmap
- [Throughput vs Latency](08-run-monitor-maintain/monitoring/throughput-vs-latency.md) - performance tradeoffs
- [Observability and Tracing](08-run-monitor-maintain/observability-and-tracing.md) - OpenTelemetry, Prometheus/Grafana, distributed tracing for agent systems
- [Alerting and Maintenance](08-run-monitor-maintain/alerting-and-maintenance.md) - alert thresholds, deployment patterns, rollback strategies, runbook automation

### 9. Safety, Ethics, and Compliance (5%)

- [Enterprise AI Factory](09-safety-ethics-compliance/enterprise-ai-factory.md) - governance and responsible AI
- [Responsible AI and Auditing](09-safety-ethics-compliance/responsible-ai-and-auditing.md) - bias types, fairness audits, EU AI Act, compliance frameworks, red teaming

### 10. Human-AI Interaction and Oversight (5%)

- [Human-AI Interaction Overview](10-human-ai-interaction/README.md) - human-in-the-loop patterns, oversight, safety guardrails, UX
- [Human-AI Oversight](10-human-ai-interaction/human-ai-oversight.md) - HITL workflows, escalation protocols, transparency and explainability, trust calibration, accessibility
- See also: [Error Handling](01-agent-architecture-and-design/error-handling.md) for failure modes and recovery patterns, [Human-Agent UI Design](01-agent-architecture-and-design/human-agent-ui-design.md) for interface patterns

## Reference

- [Terms](terms.md) - Glossary of agentic AI and NVIDIA-specific terminology
- [Resources](Resources.md) - Official NVIDIA documentation, research papers, and external links

## How to Use These Notes

1. **Read sequentially** within each domain to build foundational understanding
2. **Cross-reference** between architecture patterns and NVIDIA implementation examples
3. **Study frameworks** alongside architecture concepts to see practical application
4. **Review NVIDIA technologies** to understand the deployment and optimization layer
5. **Use as reference** during exam practice and real-world system design

## Key Takeaways

- **Agent Architecture** is foundational-understand reactive vs. deliberative systems before adding complexity
- **Checkpointing & Memory** are critical for production reliability and cost efficiency
- **NVIDIA's Stack** (NIMs, Triton, NeMo) enables scalable deployment of agentic systems
- **Framework choice** depends on workflow complexity-LangGraph for branching logic, CrewAI for role-based teams, AutoGen for multi-agent conversations
- **Error handling** and **evaluation** patterns distinguish production systems from prototypes

## Exam Preparation Tips

1. **Focus on high-weight domains first** - Agent Architecture (15%), Agent Development (15%), Deployment (13%), Evaluation (13%) account for 56% of the exam
2. **Understand NVIDIA's competitive advantages** - Why NIMs over APIs? Why Triton for inference? How does NeMo integrate with agent frameworks?
3. **Connect theory to practice** - Every architecture pattern should map to a framework implementation and a deployment scenario
4. **Practice with real code** - The examples in this repo are production-ready; use them to build intuition
5. **Know the trade-offs** - Latency vs throughput, cost vs performance, reliability vs speed in agent design
6. **Study for application** - Questions test practical judgment (which technology to use when) not just definitions

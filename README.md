# NVIDIA Certified Professional: Agentic AI (NCP-AAI)

Comprehensive exam notes and study resources for the [NVIDIA Agentic AI Professional Certification](https://www.nvidia.com/en-us/learn/certification/agentic-ai-professional/).

## About This Repository

This repository consolidates knowledge and best practices for building production-grade agentic AI systems on NVIDIA's platform. It covers foundational concepts through advanced enterprise deployment patterns, with emphasis on NVIDIA-specific technologies like NIMs, NeMo, Triton, and NeMo Agents.

**Who should use this?**
- Developers and architects preparing for the NCP-AAI certification
- Engineers building agent-based AI systems
- Teams deploying agentic AI at scale on NVIDIA infrastructure
- Anyone seeking deep understanding of agent architectures and NVIDIA's agentic AI stack

**What you'll find here:**
- **Agent Architecture Foundations** — Core concepts, design patterns (reactive, deliberative, hybrid, multi-agent)
- **Reasoning & Problem-Solving** — ReAct, Chain-of-Thought, Tree of Thoughts, self-reflection techniques
- **Production Patterns** — Error handling, memory management, checkpointing, evaluation frameworks
- **NVIDIA Ecosystem Deep Dives** — NIMs, NeMo, Triton, performance optimization, enterprise deployment
- **Framework Comparisons** — LangChain/LangGraph, CrewAI, AutoGen with NVIDIA integration examples
- **Advanced Topics** — RAG, vector databases, tool integration, multi-agent coordination

The notes are structured to mirror exam domains while maintaining practical utility for implementation.

## Exam Blueprint & Domain Coverage

The NCP-AAI exam covers **10 weighted domains** across 60–70 questions (120-minute time limit):

| Domain | Weight | Study Resources |
| --- | --- | --- |
| **Agent Architecture and Design** | 15% | Agent Architecture Overview, 4-Layer Architecture, Reasoning Patterns |
| **Agent Development** | 15% | LangChain, CrewAI, AutoGen, Code Examples |
| **Evaluation and Tuning** | 13% | Evaluation and Tuning Techniques, Frameworks |
| **Deployment and Scaling** | 13% | Triton, Enterprise Deployment Pattern, Platform Components |
| **Cognition, Planning, and Memory** | 10% | Reasoning Patterns, Memory & State Management, LangGraph Checkpoints |
| **Knowledge Integration and Data Handling** | 10% | RAG and Tools, Vector DBs, Embeddings |
| **NVIDIA Platform Implementation** | 7% | NIMs, NeMo, Triton, AIQ Toolkit, Architecture Patterns |
| **Run, Monitor, and Maintain** | 5% | Triton Performance Analyzer, Enterprise Deployment Pattern |
| **Safety, Ethics, and Compliance** | 5% | Error Handling, Enterprise AI Factory |
| **Human-AI Interaction and Oversight** | 5% | Error Handling, Multi-Agent Coordination patterns |

## Quick Start

**For exam preparation:**
1. Master **Agent Architecture and Design** (15%) — foundation for all other domains
2. Study **Agent Development** (15%) using the framework guides
3. Deep-dive into **Cognition, Planning, Memory** (10%) and **Knowledge Integration** (10%)
4. Review **NVIDIA Platform Implementation** (7%) and **Deployment** (13%)
5. Cover remaining domains: Evaluation, Monitoring, Safety, Human-AI Interaction

**For implementation:** Use the Framework guides as starting points, reference NVIDIA technologies for infrastructure decisions, and apply patterns from Error Handling and enterprise deployment sections.

## Study Materials by Exam Domain

### 1. Agent Architecture and Design (15%)

- [Agent Architecture Overview](01-agent-architecture-and-design/agent-architecture-overview.md) — reactive, deliberative, hybrid, learning, multi-agent, tool-augmented architectures
- [4 Layer Agent Architecture](01-agent-architecture-and-design/4-layer-agent-architecture.md) — perception, cognition, planning, execution
- [Reasoning Patterns](01-agent-architecture-and-design/reasoning-patterns.md) — ReAct, Chain-of-Thought, Tree of Thoughts, Self-Reflection
- [Error Handling](01-agent-architecture-and-design/error-handling.md) — resilience patterns and fault tolerance

### 2. Agent Development (15%)

- [Framework Comparison](02-agent-development/agent-orchiestration-frameworks.md) — AutoGen, LangGraph, CrewAI, LangChain overview
- [LangChain/LangGraph](02-agent-development/langchain.md) — stateful graphs, memory, checkpoints, NVIDIA integration
- [CrewAI](02-agent-development/crewai.md) — role-based agent teams
- [AutoGen](02-agent-development/autogen.md) — multi-agent conversations

### 3. Evaluation and Tuning (13%)

- [Evaluation and Tuning Techniques](03-evaluation-and-tuning/evaluation-and-tuning-techniques.md)

### 4. Deployment and Scaling (13%)

**Inference & Optimization:**
- [Triton Inference Server](04-deployment-and-scaling/inference/triton-inference-server.md) — high-performance inference
- [Triton Performance Analyzer](04-deployment-and-scaling/inference/triton-performance-analyzer.md) — profiling and optimization

**Enterprise Deployment:**
- [Enterprise Deployment Pattern](04-deployment-and-scaling/deployment/enterprise-deployment-pattern.md)

### 5. Cognition, Planning, and Memory (10%)

*Study materials coming soon* — See reasoning patterns and memory sections in Agent Architecture and LangGraph documents

### 6. Knowledge Integration and Data Handling (10%)

- [RAG and Tools](06-knowledge-integration-data/rag-and-tools.md) — embeddings, routers, chunking, retrieval methods, vector DBs, evaluation, advanced RAG patterns

### 7. NVIDIA Platform Implementation (7%)

- [NIMs](07-nvidia-platform-implementation/nims.md) — NVIDIA Inference Microservices
- [NeMo](07-nvidia-platform-implementation/nemo.md) — large language models
- [NeMo Agent Toolkit](07-nvidia-platform-implementation/nemo-agent-toolkit.md)
- [AIQ Toolkit](07-nvidia-platform-implementation/aiq-toolkit.md)
- [Platform Components](07-nvidia-platform-implementation/platform-components.md)
- [Architecture Patterns](07-nvidia-platform-implementation/architecture-patterns.md)
- [NVIDIA Role in Ecosystem](07-nvidia-platform-implementation/nvidia-role-in-ecosystem.md)

### 8. Run, Monitor, and Maintain (5%)

- [Throughput vs Latency](08-run-monitor-maintain/monitoring/throughput-vs-latency.md) — performance tradeoffs
- [NIM vs NeMo](08-run-monitor-maintain/monitoring/nim-vs-nemo.md) — deployment and operation considerations

### 9. Safety, Ethics, and Compliance (5%)

- [Enterprise AI Factory](09-safety-ethics-compliance/enterprise-ai-factory.md) — governance and responsible AI

### 10. Human-AI Interaction and Oversight (5%)

*Study materials coming soon* — See error handling and multi-agent patterns

## Reference

- [Terms](terms.md) — Glossary of agentic AI and NVIDIA-specific terminology
- [Resources](Resources.md) — Official NVIDIA documentation, research papers, and external links

## How to Use These Notes

1. **Read sequentially** within each domain to build foundational understanding
2. **Cross-reference** between architecture patterns and NVIDIA implementation examples
3. **Study frameworks** alongside architecture concepts to see practical application
4. **Review NVIDIA technologies** to understand the deployment and optimization layer
5. **Use as reference** during exam practice and real-world system design

## Key Takeaways

- **Agent Architecture** is foundational—understand reactive vs. deliberative systems before adding complexity
- **Checkpointing & Memory** are critical for production reliability and cost efficiency
- **NVIDIA's Stack** (NIMs, Triton, NeMo) enables scalable deployment of agentic systems
- **Framework choice** depends on workflow complexity—LangGraph for branching logic, CrewAI for role-based teams, AutoGen for multi-agent conversations
- **Error handling** and **evaluation** patterns distinguish production systems from prototypes

## Exam Preparation Tips

1. **Focus on high-weight domains first** — Agent Architecture (15%), Agent Development (15%), Deployment (13%), Evaluation (13%) account for 56% of the exam
2. **Understand NVIDIA's competitive advantages** — Why NIMs over APIs? Why Triton for inference? How does NeMo integrate with agent frameworks?
3. **Connect theory to practice** — Every architecture pattern should map to a framework implementation and a deployment scenario
4. **Practice with real code** — The examples in this repo are production-ready; use them to build intuition
5. **Know the trade-offs** — Latency vs throughput, cost vs performance, reliability vs speed in agent design
6. **Study for application** — Questions test practical judgment (which technology to use when) not just definitions

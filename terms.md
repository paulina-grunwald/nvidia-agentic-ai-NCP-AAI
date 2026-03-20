# NVIDIA Agentic AI Certification - Key Terms & Definitions

Organized by topic for exam preparation and quick reference.

---

## 1. Core LLM & Tokenization

### Token
The smallest linguistic unit that LLMs use to break down and process natural language. Tokens can be words, subwords, or characters. Core to LLM inference performance metrics. A word typically breaks into 1-2 tokens, with average ~4 chars per token.

### Tokenization
The process of breaking down text into smaller units (tokens). Uses exact pattern matching - does not recognize synonyms or related words as similar, and lacks understanding of context and meaning.

### Input Sequence Length (ISL)
The number of tokens that the LLM receives as input from the user. Part of the total context window utilization.

### Output Sequence Length (OSL)
The number of tokens the LLM generates as output. Impacts latency and computational cost (generation is more expensive than input processing).

### Context Length / Context Window
The maximum number of tokens an LLM can process at each generation step, including both input tokens and output tokens generated so far. Modern LLMs range from 4K to 128K+ tokens. Larger context windows enable handling longer documents but increase memory and compute requirements.

### Attention Mechanism
A neural network technique that weights the importance of relationships between tokens in an input sequence. Self-attention allows the model to examine the entire sequence simultaneously and focus on specific tokens. Core breakthrough of Transformer architecture - enables capturing global dependencies and contextual understanding of tokens relative to all other tokens.

### Transformer Architecture
Neural network architecture based on multi-head attention mechanisms. Replaced sequential processing with parallel attention, allowing simultaneous processing of entire sequences. Foundation of modern LLMs (GPT, Claude, LLaMA).

### LSTM (Long Short-Term Memory)
A type of recurrent neural network (RNN) architecture designed to learn long-range dependencies in sequential data. Uses gates (forget, input, output) to control what information to retain or discard over time steps. Predecessor to Transformers for sequence modeling. Limited by sequential processing (slower than Transformers) but foundational for understanding memory in neural networks.

### MLP (Multi-Layer Perceptron)
The simplest form of a feedforward neural network — just stacked layers of neurons (input → hidden → output) with activation functions between them. No recurrence, no attention — just dense/fully-connected layers. Used as building block in larger architectures and for simple regression/classification tasks. Limited by inability to capture sequential or contextual patterns.

---

## 2. Embeddings & Semantic Understanding

### Embeddings
Numerical representations (vectors) of data that convert complex, high-dimensional information into low-dimensional vectors. Enable machines to process and analyze data efficiently while capturing semantic relationships - unlike tokens, embeddings understand meaning and similarity.

### Vector Database / Vector Store
Specialized database optimized for storing and retrieving embeddings. Examples: Milvus, Pinecone, Weaviate, Qdrant, Supabase. Used in RAG systems to store document embeddings for semantic search.

### Semantic Search
Retrieval based on meaning and context rather than exact keyword matching. Finds conceptually similar content by comparing embeddings. More intelligent than keyword search for understanding intent.

### Embedding Model
Neural network trained to convert text/data into embeddings. Examples: NVIDIA nv-embed-v2, OpenAI's text-embedding-3, sentence-transformers. Different models capture different types of semantic relationships.

---

## 3. Agent Architecture & Frameworks

### Agent
An autonomous system that uses an LLM as its reasoning engine to:
1. Perceive input/environment
2. Reason about the task
3. Make decisions
4. Take actions (via tools)
5. Observe results
6. Iterate until task complete

### ReAct (Reasoning + Acting)
Agent reasoning pattern that alternates between:
- **Thought**: Internal reasoning step
- **Action**: Tool call or external action
- **Observation**: Result feedback
Enables self-correction through explicit reasoning traces. Improves explainability and task success rate vs. pure generation.

### Chain-of-Thought (CoT)
Prompting technique where the LLM generates intermediate reasoning steps before the final answer. Improves accuracy on multi-step problems. Can be zero-shot ("think step by step") or few-shot (examples provided).

### Tree-of-Thoughts (ToT)
Advanced reasoning pattern that explores multiple reasoning branches simultaneously, enabling backtracking and reconsidering approaches. More comprehensive than CoT but computationally expensive.

### Self-Reflection
Agent capability to review its own outputs, identify errors, and correct them. Reduces hallucination and improves quality. Can be triggered after each response or periodically.

### LangChain
Open-source Python framework for building agent applications. Provides abstractions for: LLM calls, prompt templates, memory management, tool integration, chains (sequences of operations). Integrates with 100+ LLM providers and tools.

### CrewAI
Framework for multi-agent systems. Emphasizes agent roles, collaboration, and delegation. Better suited than LangChain for complex multi-agent coordination tasks.

### AutoGen (Microsoft)
Multi-agent conversation framework. Agents communicate via conversation turns, enabling complex team dynamics and hierarchical reasoning. Good for workflow automation.

### LangGraph
LangChain's stateful graph-based agent framework. Enables:
- Explicit control flow (branching, loops, conditionals)
- Persistent checkpointing (save/resume execution)
- Long-running agents
- Complex agent workflows

### Model Context Protocol (MCP)
Standard protocol for agent tool integration (language-agnostic, framework-agnostic). Defines how agents discover, describe, and invoke tools. Replaces ad-hoc tool integration with standardized approach.

---

## 4. Memory Systems

### Conversation Memory / Turn History
Stores recent interaction history in sequence. Provides immediate context for continuity across turns. Size-limited to manage context window usage. Example: last 10-20 conversation turns.

### Entity Memory
Extracts and maintains structured facts about important entities (customers, orders, issues, etc.). Updated automatically during conversation. Provides focused, factual context without full history. Reduces tokens needed vs. conversation history.

### Vector Memory / Semantic Memory
Stores past interactions as embeddings for semantic search. Retrieves contextually similar past interactions based on current query. Enables learning from patterns across many past interactions.

### Episodic Memory
Long-term memory of specific past events/interactions. Used for case-based reasoning ("this is like when..."). Different from semantic memory which captures generalizations.

### Memory Provider
Component that manages a specific memory type. NAT (NeMo Agent Toolkit) provides built-in memory providers for conversation, entity, and vector memories. Can be composed for multi-layered memory.

### State Persistence / Checkpointing
Saving agent execution state (memory, context, intermediate results) to external storage (DB, file). Enables resuming interrupted execution. LangGraph uses PostgreSQL for checkpointing. Critical for long-running agents.

---

## 5. Retrieval-Augmented Generation (RAG)

### RAG (Retrieval-Augmented Generation)
Technique that combines:
1. Retrieval: Find relevant documents from knowledge base
2. Augmentation: Include retrieved docs in context
3. Generation: LLM generates answer grounded in retrieved context

Reduces hallucination vs. pure generation. Foundation of knowledge-aware agents.

### RAGAS Framework
**Reference-free evaluation framework for RAG systems.** The standard for evaluating both components of RAG pipelines across two dimensions:

**Retrieval Metrics** (Are we getting the right documents?):
- **Context Recall**: Percentage of relevant documents actually retrieved. Identifies under-retrieval (missing information).
- **Context Precision**: Percentage of retrieved documents that are relevant/focused. Identifies over-retrieval (irrelevant noise).

**Generation Metrics** (Given context, is the LLM generating good answers?):
- **Faithfulness**: Is generated answer grounded in retrieved context with no hallucinations?
- **Answer Relevance**: Does answer directly address the input question?

Advantages: No need for human ground truth labels; uses LLM-based evaluation. Separates evaluation of retriever vs. generator to identify bottlenecks. Enables root cause analysis (retrieval problem vs. generation problem). [Read more](https://docs.ragas.io/en/latest/concepts/metrics/)

### Reranker
Model that re-ranks initial retrieval results. Two-pass retrieval:
1. First pass: Fast approximate search (return top 20)
2. Second pass: Reranker ranks by relevance (return top 5)

Improves precision at small K. Adds latency but increases quality.

### Top-K Retrieval
Returns K most similar documents from vector DB. Trade-off: larger K = better recall but more tokens in context and more latency.

### Chunk Size
Length of text segments stored in vector DB (typically 256-512 tokens). Smaller chunks = more granular retrieval but more total chunks. Affects both quality and cost.

---

## 6. Inference & Optimization

### Inference Performance
How FAST the model responds. Measured by:
- **Latency**: Time to first token (TTFT), time per token (TpT), end-to-end latency
- **Throughput**: Requests/second, tokens/second
- **Cost**: Cost per request, cost per token

Different from reasoning performance (quality).

### Throughput
Number of requests/tokens processed per second. Limited by GPU memory, compute capacity, and batch size. Higher throughput = higher GPU utilization.

### Latency
Time required to generate a response. Measured in multiple ways:
- **Time-to-first-token (TTFT)**: How long before first token appears
- **Time-per-token (TpT)**: Average time per subsequent token
- **End-to-end latency**: Total time from request to complete response

Affected by model size, input length, batch size, and optimization level.

### Quantization (INT8 / FP8)
Technique reducing model size by converting floating-point weights to lower precision (e.g., FP32 → INT8 or FP8). Two main approaches differ in precision handling:
- **INT8 (8-bit integer)**: Uses uniform spacing across 256 values. Reduces memory by 75%, speeds computation, with <1% accuracy loss. Struggles with outlier values (extreme weights/activations) that either get clipped or force range stretching.
- **FP8 (8-bit floating-point)**: Retains floating-point structure (sign, exponent, mantissa) with flexible range via exponent. Handles outliers better than INT8, supports wider dynamic range. Superior accuracy across workloads (92.64% coverage vs 65.87% for INT8) but uses 40-50% more silicon than INT8.

Two approaches: Post-Training Quantization (PTQ) or Quantization-Aware Training (QAT). Production systems typically use mixed precision (INT8 + FP8) tuned per layer. [More info](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)

### Tensor Parallelism
Distributing a single model across multiple GPUs by splitting individual parameter tensors (intra-layer parallelism). Each GPU holds 1/N chunks of weights and performs computation on partial tensors. Unlike pipeline parallelism which keeps weights intact but partitions layers, tensor parallelism splits individual weights across devices. Benefits: reduces model state and activation memory, avoids pipeline bubble problem (all devices work on same batch simultaneously). Drawbacks: requires high-speed interconnects (NVLink) for frequent device communication, typically restricted to within-node GPUs. Essential for massive models where a single parameter (e.g., large embedding table, softmax layer) exceeds GPU memory. [More info](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html)

### Pipeline Parallelism
Splitting model layers across multiple GPUs, with different batches processing different stages simultaneously. Reduces idle time vs. tensor parallelism. Good for long models but adds complexity.

### KV Cache / Paged Attention
**KV Cache**: Pre-computes and caches Key-Value embeddings from previous tokens during decoding, avoiding recomputation on subsequent tokens. Stored in GPU memory.

**Paged Attention** (vLLM): Advanced KV cache optimization inspired by OS virtual memory/paging. Breaks memory into fixed-size blocks (~16 tokens per block) and maintains logical→physical block mappings, allowing non-contiguous block placement. Solves the KV cache fragmentation problem—existing systems waste 60-80% of KV cache memory, while Paged Attention achieves <4% waste. Enables larger batch sizes and higher throughput. Key innovation: blocks are allocated on-demand as tokens generated, making memory layout flexible. Deployed in vLLM, delivering up to 24x higher throughput vs. HuggingFace Transformers. Major contributor to speed improvements (2-3x). [More info](https://blog.vllm.ai/2023/06/20/vllm.html)

### Batch Size
Number of requests processed simultaneously. Larger batches = better GPU utilization and throughput, but higher latency per individual request. Trade-off depends on use case.

### Dynamic Batching
Automatically batching requests of varying lengths. Reduces padding overhead and improves throughput compared to fixed batch sizes.

### NVIDIA Inference Microservices (NIMs)
Pre-optimized containerized inference engines. Pre-configured with TensorRT-LLM optimizations, tensor parallelism, dynamic batching. Turn-key deployment. Alternative to manually optimizing with TensorRT-LLM.

### TensorRT-LLM
NVIDIA's open-source library for optimizing LLM inference. Provides:
- Quantization (INT8, FP8)
- KV cache optimization
- Custom CUDA kernels
- Multi-GPU serving
- Paged attention

Produces 2-3x speedup vs. naive inference. Requires more setup than NIMs but maximum control.

### Custom CUDA Kernels
Hand-optimized GPU compute kernels written in CUDA to accelerate specific operations beyond standard libraries (cuBLAS, cuDNN). Key optimization techniques: kernel fusion (combining multiple operations), memory coalescing (sequential memory access), shared memory usage for temporary storage, minimizing branch divergence. Enable operation-specific tuning for target hardware. Applications: fused attention computations, optimized matrix-vector operations, specialized RNN/LSTM implementations. Performance gains: specialized kernels achieve 7.3x speedup over generic implementations (e.g., PyTorch cuBLAS) through memory hierarchy exploitation. Used extensively in TensorRT-LLM and inference engines for latency-critical deployments. Trade-off: requires significant engineering effort but highest possible optimization for known workloads. [More info](https://huggingface.co/blog/kernel-builder)

### Triton Inference Server
Multi-model serving platform. Manages multiple models on shared infrastructure, handles batching, scheduling, and load balancing. Useful when deploying LLM + embeddings + reranker.

---

## 7. Evaluation & Quality

### Reasoning Performance
How WELL the model thinks/reasons. Measured by:
- Task success rate
- Accuracy on complex tasks
- Quality of intermediate reasoning
- Tool selection accuracy

Different from inference performance (speed).

### Ground Truth
Verified, correct answer or reference standard used to evaluate model performance. The "answer key" against which you measure predictions. Required for supervised evaluation but not needed for reference-free methods like RAGAS.

### Hallucination
False claims in model output that are not grounded in provided context or knowledge. A major quality issue for RAG and knowledge-grounded agents. Detected via:
- Entailment checking against knowledge base
- Semantic consistency verification
- Explicit contradiction detection

### Hallucination Detection
Automated evaluation of whether model claims are verifiable. Methods:
- **Entailment-based**: Use NLI model to check if claim follows from context
- **Knowledge-base grounding**: Check if claim exists in knowledge base
- **Semantic consistency**: Verify internal consistency of claims

Measured as hallucination rate (% of responses with ungrounded claims).

### Task Success Rate
Percentage of user tasks completed correctly. Binary metric (succeeded or failed). Depends heavily on task definition and evaluation criteria.

### Semantic Similarity
Cosine similarity between embeddings of two texts. Ranges 0-1. Measures how conceptually similar content is. Used in evaluation and RAG retrieval.

### Faithfulness (RAGAS)
Measure of whether generated answer is grounded in provided context. No hallucinations = high faithfulness. Key dimension of RAG quality.

### Answer Relevance (RAGAS)
Measure of whether generated answer directly addresses the input question. Evaluates answer-question alignment. Key dimension of RAG quality.

### Context Precision (RAGAS)
Measure of whether retrieved context is relevant/focused. Precision of retriever. Identifies over-retrieval (too much noise).

### Context Recall (RAGAS)
Measure of whether all relevant context documents were retrieved. Recall of retriever. Identifies under-retrieval (missing information).

---

## 8. Model Development & Fine-tuning

### Fine-tuning
Adapting a pre-trained model to a specific task/domain. Cheaper than training from scratch. Quality depends on data quality and approach chosen.

### PEFT / LoRA (Parameter-Efficient Fine-Tuning)
Fine-tuning only 0.1% of parameters via low-rank adapters.
- **Cost**: 1 GPU, 1-2 hours
- **Quality**: 90% of full fine-tuning quality
- **Best for**: Fast iteration, pilots, resource-constrained settings

### LoRA (Low-Rank Adaptation)
A parameter-efficient fine-tuning (PEFT) method that adapts pre-trained models by training small low-rank "adapter" matrices while keeping original weights frozen. Works by decomposing weight updates into two smaller matrices (A and B) that approximate weight changes. Injects trainable rank decomposition matrices into each Transformer layer. Reduces trainable parameters to ~0.1% of full model (e.g., GPT-3 reduced to 18M trainable parameters), lowering GPU memory requirements by ~2/3 vs. full fine-tuning. No additional inference overhead. [More info](https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html)

### DoRA (Weight-Decomposed Low-Rank Adaptation)
An enhancement to LoRA that decomposes pre-trained weights into **magnitude** and **direction** components, then applies LoRA only to the directional updates. The magnitude component is separately fine-tuned while direction is updated via LoRA for efficiency. Improves upon LoRA by enhancing both learning capacity and training stability. Consistently outperforms LoRA on tasks like commonsense reasoning, visual instruction tuning, and multimodal understanding. No additional inference overhead vs. LoRA. [More info](https://sebastianraschka.com/blog/2024/lora-dora.html)

### Supervised Fine-Tuning (SFT)
Full model fine-tuning on labeled examples (question-answer pairs, instructions, etc.).
- **Cost**: 8 GPUs, 1 day
- **Quality**: Excellent for task-specific performance
- **Best for**: Production deployment

### DPO (Direct Preference Optimization)
Fine-tuning using preference pairs (good response vs. bad response) without explicit reward model. Improves alignment and reduces hallucination.
- **Cost**: 4 GPUs, 12 hours
- **Quality**: Very high alignment
- **Best for**: Alignment-critical tasks (safety, compliance)

### NeMO Framework
NVIDIA's open-source framework for training and fine-tuning LLMs. Provides:
- Distributed training utilities
- PEFT/LoRA implementations
- DPO support
- Export to TensorRT-LLM for optimization

### NeMO Curator
Data pipeline for cleaning and preparing training data:
- Quality filtering (removes low-quality documents)
- Deduplication (removes near-duplicates)
- PII redaction (protects privacy)
- Language filtering
- Toxic content filtering

High-quality data is critical for fine-tuning success.

---

## 9. Data Management & Quality Monitoring

### Overfitting
When a model learns training data too well, capturing noise and specific patterns rather than generalizing. Results in: excellent training performance but poor performance on unseen production data. Avoided by: regularization, early stopping, diverse training data.

### Data Drift / Feature Drift
Changes in distribution of input features between training and production. Mapping from features to labels stays the same. **Example**: Training on 90% English queries, production has 60% Spanish.

*Note: Different from Concept Drift (relationship changes).*

[More details](https://www.dataversity.net/articles/data-drift-vs-concept-drift-what-is-the-difference/)

### Concept Drift
Evolution of data that changes the relationship between inputs and outputs. The mapping itself changes, not just the input distribution. **Example**: Fraud detection patterns change as fraudsters adapt. **Causes**: External real-world changes that invalidate assumptions.

*Note: Different from Data Drift (distribution only changes).*

[More details](https://www.evidentlyai.com/ml-in-production/concept-drift)

### Label Drift
Distribution of model outputs changes over time. Detected by monitoring output class distributions. May indicate concept drift or model degradation.

### Drift Detection Methods
Statistical tests to identify distribution changes:
- **Kolmogorov-Smirnov (KS) test**: Compares CDFs of reference vs. production data
- **Chi-Square test**: For categorical distributions
- **Population Stability Index (PSI)**: Measures shift in feature distribution
- **Threshold-based**: Alert when p-value < significance threshold (e.g., 0.05)

### Distribution Shift
Broader term for any change in data distribution (includes data drift, label drift, feature drift). Causes model performance degradation. Detected via statistical tests or monitored quality metrics.

---

## 10. Deployment & Infrastructure

### Kubernetes / K8s
Container orchestration platform for deploying and scaling applications. Features:
- **Pods**: Smallest deployable unit (container wrapper)
- **Deployments**: Manage replica sets and updates
- **Services**: Load balancing and networking
- **StatefulSets**: For stateful applications needing persistent identity

Standard for production agent deployment.

### Docker
Containerization technology. Packages application + dependencies into lightweight image. Enables reproducible deployment across environments.

### Load Balancer
Routes incoming requests to multiple agent instances. Provides:
- **Health checks**: Remove failed instances
- **Round-robin or intelligent routing**: Distribute load
- **Auto-failover**: Switch to healthy instances if one fails

Enables horizontal scaling and high availability.

### Horizontal Scaling
Increasing capacity by adding more instances/replicas. Stateless design enables simple scaling. Alternative to vertical scaling (bigger hardware).

### Vertical Scaling
Increasing capacity of existing instance (more CPU, GPU, memory). Limited by hardware ceiling. More expensive than horizontal scaling at large scale.

### High Availability (HA)
System continues operating despite component failures. Achieved via:
- Multiple replicas
- Distributed infrastructure
- Health checks and auto-failover
- Shared state (databases, caches)

SLA targets: 99.5%+ uptime.

### Service Level Agreement (SLA)
Contractual availability/performance target. **Example**: 99.5% uptime = 22 minutes downtime/month allowed. Drives architecture decisions and monitoring.

### Circuit Breaker
Fault tolerance pattern. Detects failures and stops sending requests to failing service, preventing cascading failures. Automatically recovers when service healthy.

---

## 11. Safety, Compliance & Monitoring

### Hallucination Rate
Percentage of agent responses containing false/ungrounded claims. Lower is better. Target: <5% for production. Monitored via AIQ Toolkit.

### Fairness / Demographic Parity
Metric ensuring model treats protected groups equally. Calculated as: disparate impact ratio = (minority success rate) / (majority success rate). Target: ratio > 0.8 (81% rule in lending).

### Bias Detection
Identifying systematic disparities in model behavior across protected groups (gender, age, race, etc.). Methods:
- Compare metrics by group
- Check for statistically significant differences
- Monitor for intersectional bias (combinations of attributes)

### PII (Personally Identifiable Information)
Sensitive data that can identify individuals: SSN, email, phone, credit card, medical ID. Agents must not leak PII in responses. Monitored via pattern matching and regex.

### GDPR Compliance
**General Data Protection Regulation** requirements for EU data:
- **Explainability**: Explain decisions to users ("right to explanation")
- **User Consent**: Users must consent to processing
- **Right to Deletion**: Support deleting user data
- **Audit Trail**: Log all processing for accountability

### Human-in-the-Loop (HITL)
Integrating human oversight into agent decision-making:
- **ALWAYS**: Human approves every action (high-stakes decisions)
- **ESCALATE**: Agent acts but escalates exceptions to human review (most common)
- **TERMINATE**: Agent acts, human can terminate if needed
- **NEVER**: Fully autonomous (low-risk, repetitive tasks)

### Guardrails
Constraints that limit what an agent can do:
- Action whitelist/blacklist
- Token budget limits
- Rate limits
- Input/output validation
- Prompt injection detection

Prevent harmful agent behavior.

---

## 12. NVIDIA Platform Components

### NeMo Agent Toolkit (NAT)
Framework-agnostic platform for building production-ready agents. Provides:
- Unified YAML configuration
- Memory providers (conversation, entity, vector)
- Evaluation engine
- MCP server integration
- Multi-agent coordination

Works with LangChain, CrewAI, AutoGen. Simplifies agent operations.

### NVIDIA Agent Intelligence Toolkit (AIQ)
Production monitoring and governance for agents. Provides:
- Agent profiling (performance, quality, behavior, compliance)
- Data drift detection (concept, label, feature)
- Quality metrics (hallucination, relevance, accuracy)
- Compliance monitoring (fairness, PII, GDPR, financial services)
- Production dashboards

Continuous evaluation without manual monitoring.

### NVIDIA NIM (Inference Microservice)
Pre-optimized containerized inference engine. Pre-configured with:
- TensorRT-LLM optimizations
- Optimal batching and parallelism
- Dynamic scheduling
- Monitoring hooks

Available for LLMs, embeddings, rerankers, multimodal models. Turn-key deployment alternative to manual optimization.

### TensorRT-LLM
NVIDIA's optimization library for LLM inference. Provides custom CUDA kernels, quantization, KV cache optimization, multi-GPU serving. 2-3x speedup. Requires more setup than NIMs but maximum control.

---

## 13. Specialized Metrics & Concepts

### Token Per Second (TPS)
Throughput metric: tokens generated per second. Higher = faster inference. Used to compare model performance and efficiency.

### Time-to-First-Token (TTFT)
Latency metric: time from request submission to first output token. Affects perceived responsiveness. Lower is better.

### Cost Per Request
Total cost of serving one request: (infrastructure cost / requests) + (API costs). Used for ROI analysis and cost optimization.

### Model Size
Number of parameters in model. Measured in billions (B):
- 7B: Fast, lower quality
- 13B: Balanced
- 70B: High quality, slower
- 405B: State-of-art, very slow

Affects latency, quality, and cost trade-offs.

### Context Utilization
Percentage of available context window used for a request. Impacts latency (more context = slower) and quality (more context often helps).

### Tool Selection Accuracy
Percentage of time agent correctly chooses the right tool for the task. Measures reasoning quality. Target: >90% for production.

### Error Rate
Percentage of requests that fail to complete successfully. Includes timeouts, exceptions, tool failures. Target: <2% for production. Monitored continuously.

---

## Sources

- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://docs.ragas.io/en/latest/concepts/metrics/)
- [Data Drift vs Concept Drift: What Is the Difference?](https://www.dataversity.net/articles/data-drift-vs-concept-drift-what-is-the-difference/)
- [Understanding Concept Drift in ML](https://www.evidentlyai.com/ml-in-production/concept-drift)
- [IBM: What is Quantization?](https://www.ibm.com/think/topics/quantization)
- [Transformer & Attention Mechanisms - Dive Into Deep Learning](http://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)

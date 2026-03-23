# TensorRT-LLM Study Notes

## What is TensorRT-LLM?

TensorRT-LLM is an inference optimization library that converts trained LLM models into highly optimized inference engines. Think of it as taking your model and turbocharging it specifically for serving predictions, not training. It's NOT a serving framework (that's Triton) and NOT a training tool (that's NeMo). Its sole purpose is to make inference fast.

## Core Optimizations

### Mixed Precision (FP32 → FP16 → FP8)

This is a key distinction on the exam. TensorRT-LLM uses mixed precision to reduce inference latency by lowering computation precision for certain operations. Your accuracy stays intact while speed increases dramatically.

Here's the critical exam trap: NeMo also does mixed precision, but for training acceleration. TensorRT-LLM's mixed precision accelerates inference. These serve completely different purposes.

- FP32 (full precision) is slowest but most accurate
- FP16 (half precision) cuts memory and compute roughly in half
- FP8 (8-bit) gives extreme speed boosts for even larger models

When you see "reduce inference latency without accuracy loss," think TensorRT-LLM mixed precision.

### Kernel Fusion

Multiple GPU operations get combined into a single optimized kernel. Instead of calling 10 separate GPU functions and waiting for each to finish, you call one. This eliminates:

- Memory bandwidth overhead from transferring intermediate results
- Kernel launch overhead (every GPU call has scheduling costs)

Result: significant speedup with zero accuracy impact.

### In-Flight Batching

Traditional batching waits for a full batch of requests before processing. In-flight batching is smarter: process new requests as they arrive without waiting for the batch to fill. Each request can complete at its own pace.

This is different from static batching where you queue requests until batch size is reached (wasting user time).

### Quantization (INT8, FP8)

Reduce precision from FP32 to integer (INT8) or lower float (FP8) representation. This shrinks model size and memory footprint significantly, allowing you to run massive models on limited GPU memory.

The trade-off is real but manageable: typically <1% accuracy loss for 3-4x memory savings.

Example: To run a 70B parameter model on a single GPU with limited memory, you'd combine quantization with tensor parallelism.

### KV Cache Management

The KV (Key-Value) cache stores attention layer outputs to avoid recomputing them for every token. Without KV caching, generating the second token would recompute attention for the first token (wasteful). With KV cache, you retrieve those values instantly.

Special consideration: In tree-of-thought or beam search scenarios with multiple reasoning branches, each branch needs its own KV cache state. This is why KV cache becomes critical in complex inference patterns.

## Latency vs Throughput (EXAM CRITICAL)

This distinction appears on nearly every exam question. Get this wrong, get the question wrong.

**Latency**: Time for a single request to complete (milliseconds from input to output).

**Throughput**: Total requests processed per second.

### The Batch Size Trap

Increasing batch size improves throughput but WORSENS latency:

- Batch size 1: Request goes immediately, low latency, poor throughput
- Batch size 32: Request waits for batch to fill, high latency, good throughput

Question asks "reduce inference latency for a single user request"? Batch size increases are the WRONG answer.

Question asks "improve overall system throughput"? Batch size or dynamic batching becomes part of the right answer.

### Speed Optimization Cheat Sheet

- **High inference latency problem** → Use TensorRT-LLM mixed precision, kernel fusion, quantization
- **Low inference throughput problem** → Increase batch size, enable dynamic batching
- **Slow training** → Use NeMo mixed precision
- **Questions about learning rate** → Training context only, never inference

## TensorRT-LLM in the NVIDIA Stack

Think of the pipeline:

1. **Model Training** (NeMo) → Raw model checkpoint
2. **Inference Optimization** (TensorRT-LLM) → Optimized inference engine
3. **Serving** (Triton) → Deployed on production servers
4. **Containerization** (NIM) → Packaged microservice

TensorRT-LLM handles step 2 exclusively. Don't confuse it with the other components.

## Tensor Parallelism

Split a large model across multiple GPUs. If your 70B model doesn't fit on a single A100, use tensor parallelism to distribute it across 2, 4, or 8 GPUs.

Often combined with quantization for extreme cases: large model + limited GPU memory = quantization + tensor parallelism together.

---

## Practice MCQs

**Question 1**: Your production system needs to serve inference requests with 50ms latency per request. Currently, requests take 200ms. What's the most effective optimization?

A) Increase batch size to 64
B) Apply TensorRT-LLM mixed precision and kernel fusion
C) Add more GPU servers to handle throughput
D) Switch from FP32 to NeMo mixed precision

**Answer**: B (TensorRT-LLM optimizations reduce per-request latency, while batch size would increase latency)

---

**Question 2**: Your inference system processes 10 requests per second but requires 100 requests per second. Latency is already acceptable at 50ms per request. What should you do?

A) Enable TensorRT-LLM FP8 quantization
B) Increase batch size and enable dynamic batching
C) Implement KV cache pruning
D) Use tensor parallelism across 4 GPUs

**Answer**: B (throughput is the bottleneck, not latency; batch size increases throughput)

---

**Question 3**: When should you combine quantization with tensor parallelism?

A) Always, to maximize performance
B) Only when serving training workloads
C) When your model is large and GPU memory is limited
D) Never; they conflict with each other

**Answer**: C (this combination handles size + memory constraints)

---

**Question 4**: What does KV cache store and why is it important for tree-of-thought inference?

A) Stores gradients for backpropagation
B) Stores Key-Value tensors from attention layers; each reasoning branch needs its own copy
C) Stores quantization parameters
D) Stores batch statistics for normalization

**Answer**: B (KV cache enables efficient attention reuse; multiple branches each need independent cache)

---

**Question 5**: Which statement about mixed precision is correct?

A) NeMo mixed precision speeds up training; TensorRT-LLM mixed precision speeds up inference
B) Both NeMo and TensorRT-LLM mixed precision do the same thing
C) Mixed precision always sacrifices accuracy significantly
D) TensorRT-LLM uses FP32 exclusively for accuracy

**Answer**: A (this distinction is heavily tested)
